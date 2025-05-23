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


class RouterMessagesState(TypedDict):
    """State that contains both message history and routing information."""

    messages: List[BaseMessage]
    intent: Optional[Literal["weather", "number", "file", "general"]]


def llm_node(state: RouterMessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
    # return {"messages": messages + [response]}


def should_use_tools_node(state: RouterMessagesState) -> Literal["tools", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, we end the graph execution
    return "end"


def build_state_graph():
    builder = StateGraph(RouterMessagesState)

    builder.add_node("llm", llm_node)
    tool_node = ToolNode(tools=tools)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        should_use_tools_node,
        {
            "tools": "tools",  # Route to tools node
            "end": END,  # End execution
        },
    )
    # builder.add_edge("tools", "llm")

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
    # TODO why intent none here?
    state = graph.invoke(
        {"messages": current_messages, "intent": None},
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
