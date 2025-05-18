from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs_langchain import factory
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    factory = factory.GoogleFactory()
    llm = factory.llm()
    embeddings = factory.embeddings()
    vector_store = Chroma(
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="lograg",
        embedding_function=embeddings,
        # persist_directory=None,
        persist_directory="./temp/chroma_logs_langchain",
    )

    # Index documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = []
    with open("temp/syslog", "r") as f:
        docs.append(Document(page_content=f.read(), metadata={"source": "syslog"}))

    all_splits = text_splitter.split_documents(docs)
    doc_ids = vector_store.add_documents(documents=all_splits)
    logger.info(f"Indexed {len(doc_ids)} documents into the vector store.")

    # TODO what happens if we have persistence in Chroma enabled and we keep indexing the same docs?
