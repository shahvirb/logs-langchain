from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from logs_langchain import factory, ingest, lograg, hosts, ssh
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    google_factory = factory.GoogleFactory()
    llm = google_factory.llm()
    embeddings = google_factory.embeddings()
    logger.info("LLM and embeddings initialized")
    vector_store = factory.vector_store(
        embeddings, persist_directory="./temp/chroma_logs_langchain"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Your sole purpose is to determine what server is being discussed here if any. If you don't know say 'NONE' exactly otherwise return the server name only.",
            ),
            ("user", "{user}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    original_question = "What is happening with tailscale in server openmediavault?"
    hostname = chain.invoke({"user": original_question})
    print(f"Host Identified: {hostname}")

    if hostname != "NONE":
        foundhost = hosts.HOSTS.get(hostname, None)
        if foundhost is None:
            print(f"Host {hostname} not found in HOSTS dictionary.")
        else:
            print(f"Found host: {hostname}")
            consent = input(f"Do you want to connect to {hostname}? (y/n): ").strip().lower()
            if consent != "y":
                print("Consent failed. Exiting.")
                exit(0)
            # Use the SSH client to connect to the server
            ssh_client = ssh.SSHClient(
                host=hostname,
                user=foundhost["username"],
                key_filename=foundhost["key_file"],
                logger=logger,
            )
            with ssh_client:
                # Example command to run on the remote server
                command = f"echo 'Hello from inside {hostname}!'"
                output = ssh_client.run_command(command)
                print(output)
                
        

    # Ingest files into the vector store
    # file_paths = ["./src/examples/agent.py"]
    # ingest.ingest_files(file_paths, vector_store)
    # show_vector_store_statistics(vector_store)

    # Define prompt for question-answering
    # prompt = hub.pull("rlm/rag-prompt")
    # rag_graph = lograg.RAGGraph(prompt, llm, vector_store)
    # rag_graph.compiled.invoke({"question": "What is happening with tailscale?"})
