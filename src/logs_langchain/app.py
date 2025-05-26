from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from logs_langchain import factory, prompts, tools
from typing import cast, TypedDict, List, Optional, Literal
import chainlit as cl
import logging

logger = logging.getLogger(__name__)

# TODO don't use globals, use the cl.user_session
google_factory = factory.GoogleFactory()
llm = google_factory.llm(model="gemini-2.5-flash-preview-05-20")
llm = llm.bind_tools(tools.all)


def general_chat_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
    # return {"messages": messages + [response]}


def router_tools_node(
    state: MessagesState,
) -> Literal["tools", "dangerous_command_verification", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # write a switch case here based on last_message.tool_calls[0]
        match last_message.tool_calls[0].get("name"):
            case "ssh_command":
                return "dangerous_command_verification"
            case _:
                return "tools"
    return "__end__"


def router_explain_node(state: MessagesState) -> Literal["explain", "ssh_explain"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.name == "ssh_command":
        return "ssh_explain"
    return "explain"


def explain_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    # tool_result = messages[-1].content
    # call = messages[-2]
    response = llm.invoke(messages + [prompts.explain_command_result])
    return {"messages": [response]}


def ssh_explain_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    question = messages[-3].content
    command = messages[-2].tool_calls[0].get("args").get("command")
    output = messages[-1].content

    response = llm.invoke(
        messages
        + prompts.expert_linux_debugger.format_messages(
            question=question, command=command, output=output
        )
    )
    return {"messages": [response]}


def dangerous_command_verification_node(state: MessagesState) -> MessagesState:
    messages = state["messages"]
    tool_call = messages.pop(-1)
    command = tool_call.tool_calls[0].get("args").get("command")
    parser = PydanticOutputParser(pydantic_object=prompts.DangerousCommand)
    chain = prompts.dangerous_command_verification | llm | parser
    response = chain.invoke(
        {"command": command, "format_instructions": parser.get_format_instructions()}
    )

    if not response.is_dangerous:
        return {"messages": messages + [tool_call]}
    return {
        "messages": [
            AIMessage(
                content="This command is dangerous and cannot be executed. Please modify your request."
            )
        ]
    }
    # TODO maybe we should raise a consent check flag here so that the user can still force the command


def router_after_verification(state: MessagesState) -> Literal["tools", "general_chat"]:
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message is an AIMessage saying the command is dangerous
    if isinstance(last_message, AIMessage) and "dangerous" in last_message.content:
        return "general_chat"
    # Otherwise it's the original tool call put back in messages
    else:
        return "tools"


def build_state_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("general_chat", general_chat_node)
    tool_node = ToolNode(tools=tools.all)
    builder.add_node("tools", tool_node)
    builder.add_node("explain", explain_node)
    builder.add_node("ssh_explain", ssh_explain_node)
    builder.add_node(
        "dangerous_command_verification", dangerous_command_verification_node
    )

    builder.add_edge(START, "general_chat")

    builder.add_conditional_edges(
        "general_chat",
        router_tools_node,
    )

    builder.add_conditional_edges(
        "tools",
        router_explain_node,
    )

    builder.add_conditional_edges(
        "dangerous_command_verification",
        router_after_verification,
    )

    return builder.compile()


@cl.on_chat_start
async def start_chat():
    graph = build_state_graph()
    cl.user_session.set("graph", graph)


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    # Get existing messages (if any)
    existing_messages = cl.user_session.get("messages", [])

    # Add the new message
    current_messages = existing_messages + [HumanMessage(content=message.content)]

    # Save the updated messages in the session
    cl.user_session.set("messages", current_messages)

    # Run the graph
    state = graph.invoke(
        {"messages": current_messages},
        config=RunnableConfig(callbacks=[cb], **config),
    )

    # Update session with the latest messages
    cl.user_session.set("messages", state["messages"])

    # Send the response
    if state["messages"]:
        cl_msg = cl.Message(content=state["messages"][-1].content)
        await cl_msg.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    # logging.basicConfig(level=logging.DEBUG)
    run_chainlit(__file__)
