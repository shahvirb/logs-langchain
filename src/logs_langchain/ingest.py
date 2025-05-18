from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import logging

logger = logging.getLogger(__name__)


def ingest_files(file_paths, vector_store):
    docs = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            logger.info(f"Ingesting file {f.name}")
            file_content = f.read()
            file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

            existing_docs = vector_store.get(
                where={"file_hash": file_hash}, include=["metadatas"]
            )
            if existing_docs and existing_docs.get("metadatas"):
                logger.info(f"Skipping ingestion for {f.name} with hash {file_hash}")
            else:
                md = {
                    "filepath": file_path,
                    "file_hash": file_hash,
                }
                docs.append(Document(page_content=file_content, metadata=md))
                logger.info(f"File {f.name} will be indexed")

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)
        doc_ids = vector_store.add_documents(documents=all_splits)
        logger.info(f"Indexed {len(doc_ids)} documents into the vector store")
    else:
        logger.info("No new documents to index")
