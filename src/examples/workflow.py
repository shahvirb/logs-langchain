from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from logs_langchain import factory, ingest, lograg, hosts, ssh, prompts
import logging

logger = logging.getLogger(__name__)


def get_user_consent(prompt_message):
    consent = input(f"{prompt_message} (y/n): ").strip().lower()
    if consent != "y":
        print("Consent violated. Exiting.")
        exit(0)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    google_factory = factory.GoogleFactory()
    llm = google_factory.llm()
    embeddings = google_factory.embeddings()
    logger.info("LLM and embeddings initialized")
    vector_store = factory.vector_store(
        embeddings, persist_directory="./temp/chroma_logs_langchain"
    )

    original_question = "Is the docker container caddy running in server helium?"

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

            get_user_consent(f"Do you want to connect to {hostname}?")

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
                                prompts.sysadmin_log_context_answer
                                | llm
                                | StrOutputParser()
                            )
                            # Truncate syslog_content to avoid exceeding token limit (e.g., first 10000 characters)
                            max_syslog_length = 10000
                            truncated_syslog = syslog_content[:max_syslog_length]
                            answer = followup_chain.invoke(
                                {
                                    "question": original_question,
                                    "logs": truncated_syslog,
                                }
                            )
                            print(f"Answer based on logs:\n{answer}")
                    case "run_command":
                        cmd_chain = (
                            prompts.linux_command_determination
                            | llm
                            | StrOutputParser()
                        )
                        command_answer = cmd_chain.invoke(
                            {"question": original_question}
                        )
                        print(f"Command to run: {command_answer}")
                        get_user_consent(f"Do you want to run the command?")
                        run_output = ssh_client.run_command(command_answer)
                        print(run_output)
                        
                        # Now let's ask the original question again but with the 
                        debug_chain = (
                            prompts.expert_linux_debugger
                            | llm
                            | StrOutputParser()
                        )
                        debug_answer = debug_chain.invoke({"question": original_question, "command": command_answer, "output": run_output})
                        print(f"Expert debugger answer:\n{debug_answer}")
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
