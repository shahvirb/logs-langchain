from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from logs_langchain import factory
from typing import cast
import chainlit as cl
import logging

logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start():
    google_factory = factory.GoogleFactory()
    llm = google_factory.llm()
    cl.user_session.set("llm", llm)
    logger.info("LLM initialized")


@cl.on_message
async def on_message(message: cl.Message):
    llm = cast(ChatGoogleGenerativeAI, cl.user_session.get("llm"))

    msg = cl.Message(content="")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | llm | StrOutputParser()

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
