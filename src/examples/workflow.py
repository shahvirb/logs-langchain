from langchain import hub
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from logs_langchain import factory, ingest, lograg, hosts, ssh, prompts
import logging

logger = logging.getLogger(__name__)


def get_user_consent(prompt_message):
    consent = input(f"{prompt_message} (y/n): ").strip().lower()
    if consent != "y":
        print("Consent violated. Exiting.")
        exit(0)
    return True


def handle_read_syslog(ssh_client, hostname, original_question, llm, prompts):
    remote_syslog_path = "/var/log/syslog"
    local_syslog_path = f"./temp/{hostname}_syslog"
    try:
        ssh_client.download(remote_syslog_path, local_syslog_path)
        print(f"Syslog downloaded to {local_syslog_path}")
    except Exception as e:
        print(f"Failed to download syslog: {e}")
        return
    with open(local_syslog_path, "r") as file:
        syslog_content = file.read()
    followup_chain = prompts.sysadmin_log_context_answer | llm | StrOutputParser()
    max_syslog_length = 10000
    truncated_syslog = syslog_content[:max_syslog_length]
    answer = followup_chain.invoke(
        {
            "question": original_question,
            "logs": truncated_syslog,
        }
    )
    print(f"Answer based on logs:\n{answer}")


def handle_run_command(ssh_client, original_question, llm, prompts, get_user_consent):
    cmd_chain = prompts.linux_command_determination | llm | StrOutputParser()
    command_answer = cmd_chain.invoke({"question": original_question})
    print(f"Command to run: {command_answer}")
    get_user_consent(f"Do you want to run the command?")
    run_output = ssh_client.run_command(command_answer)
    print(run_output)
    debug_chain = prompts.expert_linux_debugger | llm | StrOutputParser()
    debug_answer = debug_chain.invoke(
        {
            "question": original_question,
            "command": command_answer,
            "output": run_output,
        }
    )
    print(f"Expert debugger answer:\n{debug_answer}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    google_factory = factory.GoogleFactory()
    llm = google_factory.llm()
    embeddings = google_factory.embeddings()
    logger.info("LLM and embeddings initialized")
    vector_store = factory.vector_store(
        embeddings, persist_directory="./temp/chroma_logs_langchain"
    )

    original_question = "In the server helium go look inside the docker logs for a container named caddy and tell me if there are any errors"

    agent_id_chain = prompts.agent_identification | llm | StrOutputParser()
    agent_id = agent_id_chain.invoke({"question": original_question})
    print(f"Invoking agent: {agent_id}")
    assert agent_id != "NONE"

    server_name_parser = PydanticOutputParser(pydantic_object=ServerName)
    server_name_chain = (
        prompts.prompt_server_name_identification | llm | server_name_parser
    )
    server_name_answer = server_name_chain.invoke(
        {
            "question": original_question,
            "format_instructions": server_name_parser.get_format_instructions(),
        }
    )
    print(f"Original Question: {original_question}")
    print(f"Host Identified: {server_name_answer.name}")

    if server_name_answer.name:
        foundhost = hosts.HOSTS.get(server_name_answer.name, None)
        if foundhost is None:
            print(f"Host {server_name_answer.name} not found in HOSTS dictionary.")
            assert False
        else:
            print(f"Found host: {server_name_answer.name}")

            get_user_consent(f"Do you want to connect to {server_name_answer.name}?")

            # Use the SSH client to connect to the server
            ssh_client = ssh.SSHClient(
                host=server_name_answer.name,
                user=foundhost["username"],
                key_filename=foundhost["key_file"],
                logger=logger,
            )
            with ssh_client:
                # Run an echo command to test the connection
                command = f"echo 'Hello from inside {server_name_answer.name}!'"
                output = ssh_client.run_command(command)
                print(output)

                if output.strip():
                    match agent_id:
                        case "read_syslog":
                            handle_read_syslog(
                                ssh_client,
                                server_name_answer.name,
                                original_question,
                                llm,
                                prompts,
                            )
                        case "run_command":
                            handle_run_command(
                                ssh_client,
                                original_question,
                                llm,
                                prompts,
                                get_user_consent,
                            )
                        case _:
                            print(f"Unknown agent_id: {agent_id}")
                            assert False
