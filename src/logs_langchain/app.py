from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from logs_langchain import factory
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


tools = [get_weather, gen_number]
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


builder = StateGraph(MessagesState)

builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")
builder.add_edge("llm", "tools")
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
