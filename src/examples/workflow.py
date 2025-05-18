from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from logs_langchain import factory, ingest, lograg, hosts, ssh, prompts
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
    
    original_question = "Is the docker container plex running in server helium?"
    
    agent_id_chain = prompts.agent_identification | llm | StrOutputParser()
    agent_id = agent_id_chain.invoke({"question": original_question})
    print(f"Invoking agent: {agent_id}")
    assert agent_id != "NONE"

    server_name_chain = prompts.server_name_identification | llm | StrOutputParser()
    hostname = server_name_chain.invoke({"question": original_question})
    print(f"Original Question: {original_question}")
    print(f"Host Identified: {hostname}")

    if hostname != "NONE":
        foundhost = hosts.HOSTS.get(hostname, None)
        if foundhost is None:
            print(f"Host {hostname} not found in HOSTS dictionary.")
            assert False
        else:
            print(f"Found host: {hostname}")
            consent = (
                input(f"Do you want to connect to {hostname}? (y/n): ").strip().lower()
            )
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
                # Run an echo command to test the connection
                command = f"echo 'Hello from inside {hostname}!'"
                output = ssh_client.run_command(command)
                print(output)

                match agent_id:
                    case "read_syslog":
                        # Let's fetch syslog now using ssh_client.download
                        remote_syslog_path = "/var/log/syslog"
                        local_syslog_path = f"./temp/{hostname}_syslog"
                        try:
                            ssh_client.download(remote_syslog_path, local_syslog_path)
                            print(f"Syslog downloaded to {local_syslog_path}")
                        except Exception as e:
                            print(f"Failed to download syslog: {e}")

                        # Now  let's read the downloaded syslog file
                        with open(local_syslog_path, "r") as file:
                            syslog_content = file.read()

                            # Ask the original_question again but now with syslog content

                            followup_chain = (
                                prompts.sysadmin_log_context_answer | llm | StrOutputParser()
                            )
                            # Truncate syslog_content to avoid exceeding token limit (e.g., first 10000 characters)
                            max_syslog_length = 10000
                            truncated_syslog = syslog_content[:max_syslog_length]
                            answer = followup_chain.invoke(
                                {"question": original_question, "logs": truncated_syslog}
                            )
                            print(f"Answer based on logs:\n{answer}")
                    case "run_command":
                        pass
                    case _:
                        print(f"Unknown agent_id: {agent_id}")

    # Ingest files into the vector store
    # file_paths = ["./src/examples/agent.py"]
    # ingest.ingest_files(file_paths, vector_store)
    # show_vector_store_statistics(vector_store)

    # Define prompt for question-answering
    # prompt = hub.pull("rlm/rag-prompt")
    # rag_graph = lograg.RAGGraph(prompt, llm, vector_store)
    # rag_graph.compiled.invoke({"question": "What is happening with tailscale?"})
