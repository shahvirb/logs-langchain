from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from logs_langchain import factory
from typing import cast
import chainlit as cl
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
import logging

logger = logging.getLogger(__name__)

google_factory = factory.GoogleFactory()
llm = google_factory.llm()


def welcome(state: MessagesState):
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Look at the users greeting and respond back very rudely like you're Linus Torvalds. Stay brief, one sentence only."),
            ("user", messages[-1].content),
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    response_text = runnable.invoke({})
    response = SystemMessage(content=response_text)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("first", welcome)

builder.add_edge(START, "first")
# builder.add_edge("final", END)

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
