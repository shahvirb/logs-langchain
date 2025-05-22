from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from logs_langchain import factory, prompts
from typing import cast, TypedDict, List, Optional, Literal
import chainlit as cl
import logging

logger = logging.getLogger(__name__)

# TODO don't use globals, use the cl.user_session
google_factory = factory.GoogleFactory()
llm = google_factory.llm(model="gemini-2.5-flash-preview-05-20")


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


@tool
def gen_number(a: int, b: int) -> int:
    """Use this to get a random number between a and b."""
    import random

    return random.randint(a, b)


@tool
def read_local_file(file_path: str) -> str:
    """Use this tool to read the contents of a local file when the user asks to read a file.
    The file_path should be a valid path on the local system.
    If the file doesn't exist or can't be read, this will return an error message."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


tools = [get_weather, gen_number, read_local_file]
llm = llm.bind_tools(tools)

# def welcome(state: MessagesState):
#     messages = state["messages"]
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Look at the users greeting and respond back very rudely like you're Linus Torvalds. Stay brief, one sentence only."),
#             ("user", messages[-1].content),
#         ]
#     )
#     runnable = prompt | llm | StrOutputParser()
#     response_text = runnable.invoke({})
#     response = SystemMessage(content=response_text)
#     # TODO should we return all the messages or just the new one?
#     return {"messages": [response]}


class RouterMessagesState(TypedDict):
    """State that contains both message history and routing information."""

    messages: List[BaseMessage]
    intent: Optional[Literal["weather", "number", "file", "general"]]


def llm_node(state: RouterMessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}


# def command_determination_node(state: RouterMessagesState):
#     messages = state["messages"]
#     last_message = messages[-1]

#     # Use the linux_command_determination prompt
#     command_chain = prompts.linux_command_determination | llm | StrOutputParser()
#     command = command_chain.invoke({"question": last_message.content})

#     # Create a message with the determined command
#     command_message = AIMessage(content=f"I'll execute this command: `{command}`")
#     return {"messages": messages + [command_message], "command": command}


def intent_classifier_node(state: RouterMessagesState):
    """Determines the user's intent from their message."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, HumanMessage):
        logger.debug("Last message is not a HumanMessage. Intent: None")
        return {"intent": None}

    content = last_message.content.lower()

    # TODO this should be re-written to use the llm instead of naive keyword matching
    # Simple keyword-based routing
    intent = "general"
    if any(word in content for word in ["weather", "temperature", "forecast"]):
        intent = "weather"
    elif any(word in content for word in ["number", "random", "generate"]):
        intent = "number"
    elif any(word in content for word in ["file", "read", "open"]):
        intent = "file"
    logger.debug(f"Intent classified as '{intent}' for message: {content}")
    return {"intent": intent}


def build_state_graph():
    builder = StateGraph(RouterMessagesState)

    # Add nodes
    builder.add_node("classifier", intent_classifier_node)
    builder.add_node("llm", llm_node)
    tool_node = ToolNode(tools=tools)
    builder.add_node("tools", tool_node)

    # Set the START node
    builder.add_edge(START, "classifier")

    # Add conditional edges from classifier to appropriate nodes
    builder.add_conditional_edges(
        "classifier",
        lambda state: state["intent"],
        {
            "weather": "tools",  # Tool will handle based on tool name
            "number": "tools",  # Tool will handle based on tool name
            "file": "tools",  # Tool will handle based on tool name
            "general": "llm",
        },
    )

    # Add END edges
    builder.add_edge("tools", END)
    builder.add_edge("llm", END)

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

    # Old starts here
    # msg = cl.Message(content=message.content)
    # state = graph.invoke(
    #     {"messages": [HumanMessage(content=msg.content)]},
    #     config=RunnableConfig(callbacks=[cb], **config),
    # )
    # # Only send the last message (assistant's response)
    # if state["messages"]:
    #     cl_msg = cl.Message(content=state["messages"][-1].content)
    #     await cl_msg.send()
    # Old ends here

    # Get existing messages (if any)
    existing_messages = cl.user_session.get("messages", [])

    # Add the new message
    current_messages = existing_messages + [HumanMessage(content=message.content)]

    # Save the updated messages in the session
    cl.user_session.set("messages", current_messages)

    # Run the graph
    state = graph.invoke(
        {"messages": current_messages, "intent": None, "command": None},
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

    logging.basicConfig(level=logging.DEBUG)
    run_chainlit(__file__)
