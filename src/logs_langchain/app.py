from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from logs_langchain import factory, prompts
from typing import cast, Literal
import chainlit as cl
import logging

logger = logging.getLogger(__name__)

google_factory = factory.GoogleFactory()
llm = google_factory.llm()


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
tool_node = ToolNode(tools=tools)

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


def llm_node(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}


# def process_tool_outputs(state: MessagesState):
#     messages = state["messages"]
#     last_message = messages[-1]

#     # Check if the last message contains tool output information
#     if hasattr(last_message, "additional_kwargs") and "tool_calls" in last_message.additional_kwargs:
#         tool_calls = last_message.additional_kwargs["tool_calls"]
#         for tool_call in tool_calls:
#             # If it's the read_local_file tool, add metadata to be used by Chainlit
#             if tool_call.get("name") == "read_local_file":
#                 # Add metadata to the message that will be used by Chainlit
#                 if "metadata" not in last_message.additional_kwargs:
#                     last_message.additional_kwargs["metadata"] = {}
#                 last_message.additional_kwargs["metadata"]["file_content"] = tool_call.get("output", "")

#     return {"messages": messages}


def command_determination_node(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]

    # Check if message content indicates a command request
    if isinstance(last_message, HumanMessage) and any(
        keyword in last_message.content.lower()
        for keyword in ["run command", "execute", "linux command", "shell command"]
    ):
        # Use the linux_command_determination prompt
        command_chain = prompts.linux_command_determination | llm | StrOutputParser()
        command = command_chain.invoke({"question": last_message.content})

        # Create a message with the determined command
        command_message = AIMessage(content=f"I'll execute this command: `{command}`")
        return {"messages": messages + [command_message], "command": command}

    # If not a command request, just pass through
    return {"messages": messages, "command": None}


builder = StateGraph(MessagesState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_node("command_determination", command_determination_node)
# builder.add_node("process", process_tool_outputs)

builder.add_edge(START, "llm")
builder.add_edge("llm", "tools")
builder.add_edge("llm", "command_determination")
# builder.add_edge("tools", "process")
builder.add_edge("tools", END)

graph = builder.compile()


# @cl.on_chat_start
# async def on_chat_start():
#     google_factory = factory.GoogleFactory()
#     llm = google_factory.llm()
#     cl.user_session.set("llm", llm)
#     logger.info("LLM initialized")


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    msg = cl.Message(content=message.content)

    state = graph.invoke(
        {"messages": [HumanMessage(content=msg.content)]},
        config=RunnableConfig(callbacks=[cb], **config),
    )
    # Only send the last message (assistant's response)
    if state["messages"]:
        cl_msg = cl.Message(content=state["messages"][-1].content)
        await cl_msg.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
