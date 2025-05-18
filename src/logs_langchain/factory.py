from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import logging
import os

logger = logging.getLogger(__name__)


class GoogleFactory:
    def __init__(
        self,
        modelstr: str = "gemini-2.0-flash",
        embedding_model: str = "models/embedding-001",
    ) -> None:
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            print(
                "Please make sure you have a .env file with GOOGLE_API_KEY='YOUR_KEY' in it."
            )
            exit()
        logging.info("GOOGLE_API_KEY successfully loaded from environment.")
        self.modelstr = modelstr
        self.embedding_model = embedding_model

    def llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(model=self.modelstr)

    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return GoogleGenerativeAIEmbeddings(model=self.embedding_model)


def vector_store(emb_func, persist_directory: str = None):
    vector_store = Chroma(
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="lograg",
        embedding_function=emb_func,
        persist_directory=persist_directory,
    )
    logger.info("Chroma vector store initialized")
    return vector_store
