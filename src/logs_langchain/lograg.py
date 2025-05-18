from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs_langchain import factory
import logging
import hashlib

logger = logging.getLogger(__name__)

def show_vector_store_statistics(vector_store):
    all_docs = vector_store.get(include=["metadatas", "documents"])
    fp = set()
    for metadata in all_docs.get("metadatas", []):
        if metadata and "filepath" in metadata:
            fp.add(metadata["filepath"])
    logger.info(f"Found {len(fp)} unique filepaths across {len(all_docs.get('documents', []))} documents in the vector store.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    factory = factory.GoogleFactory()
    llm = factory.llm()
    embeddings = factory.embeddings()
    logger.info("LLM and embeddings initialized")

    vector_store = Chroma(
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="lograg",
        embedding_function=embeddings,
        # persist_directory=None,
        persist_directory="./temp/chroma_logs_langchain",
    )
    logger.info("Chroma vector store initialized")

    # Index documents
    docs = []
    with open("temp/syslog", "r") as f:
        logger.info(f"Ingesting file {f.name}")
        # calculate a hash of the file to use as metadata
        file_content = f.read()
        file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()

        # Check if the file is already in the vector store by file_hash
        existing_docs = vector_store.get(where={"file_hash": file_hash}, include=["metadatas"])
        if existing_docs and existing_docs.get("metadatas"):
            logger.info(f"Skipping ingestion for {f.name} with hash {file_hash}")
        else:
            md = {
            "filepath": "temp/syslog",
            "file_hash": file_hash,
            }
            docs.append(Document(page_content=file_content, metadata=md))
            logger.info(f"File {f.name} will be indexed")

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        doc_ids = vector_store.add_documents(documents=all_splits)
        logger.info(f"Indexed {len(doc_ids)} documents into the vector store")
    else:
        logger.info("No new documents to index")