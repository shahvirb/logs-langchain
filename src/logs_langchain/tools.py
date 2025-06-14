from langchain_core.tools import tool
from typing import Literal
from logs_langchain import ssh, hosts


def get_user_consent(prompt_message):
    consent = input(f"{prompt_message} (y/n): ").strip().lower()
    if consent != "y":
        print("Consent violated. Exiting.")
        exit(0)
    return True


@tool
def gen_number(a: int, b: int) -> int:
    """Use this to get a random number between a and b."""
    import random

    return random.randint(a, b)


@tool
def read_local_file(file_path: str) -> str:
    """Use this tool to read the contents of a local file when the user asks to read a file.
    The file_path should be a valid path on the local system.
    If the file doesn't exist or can't be read, this will return an error message."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def ping(host) -> bool:
    """Use this to ping a server. It returns True if the server is reachable, False otherwise."""
    import subprocess

    try:
        output = subprocess.check_output(["ping", "-c", "1", host])
        return True
    except subprocess.CalledProcessError:
        return False


@tool
def ssh_command(host: str, command: str) -> str:
    """Use this to run a command on a remote server via SSH. It returns a string with the command output."""
    # get_user_consent(f"Do you want to run the command '{command}' on {host}?")
    host_info = hosts.HOSTS[host]
    with ssh.SSHClient(host, host_info["username"], host_info["key_file"]) as client:
        return client.run_command(command)


all = [gen_number, read_local_file, ping, ssh_command]
