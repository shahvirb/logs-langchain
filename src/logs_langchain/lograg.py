from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs_langchain import factory, ingest
import logging

logger = logging.getLogger(__name__)


def show_vector_store_statistics(vector_store):
    all_docs = vector_store.get(include=["metadatas", "documents"])
    fp = set()
    for metadata in all_docs.get("metadatas", []):
        if metadata and "filepath" in metadata:
            fp.add(metadata["filepath"])
    logger.info(
        f"Found {len(fp)} unique filepaths across {len(all_docs.get('documents', []))} documents in the vector store."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    google_factory = factory.GoogleFactory()
    llm = google_factory.llm()
    embeddings = google_factory.embeddings()
    logger.info("LLM and embeddings initialized")
    vector_store = factory.vector_store(
        embeddings, persist_directory="./temp/chroma_logs_langchain"
    )

    ingest.ingest_files(["temp/syslog"], vector_store)

    show_vector_store_statistics(vector_store)
