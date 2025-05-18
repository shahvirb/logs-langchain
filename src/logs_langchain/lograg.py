from langchain import hub
from langgraph.graph import START, StateGraph
from logs_langchain import factory, ingest
import logging
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAGGraph:
    def __init__(self, prompt, llm, vector_store):
        self.prompt = prompt
        self.llm = llm
        self.vector_store = vector_store
        self.compiled = self.make_graph()

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def make_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()


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

    prompt = hub.pull("rlm/rag-prompt")
    graph = RAGGraph(prompt, llm, vector_store)
    response = graph.compiled.invoke(
        {
            "question": "Summarize tailscale related lines. Also what are the log times of those related lines?"
        }
    )
    # print(response["context"])
    print("-------------")
    print(response["answer"])
