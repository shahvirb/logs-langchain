from fabric import Connection
import logging
import os
import math
import shutil
import shutil
import shutil

logger = logging.getLogger(__name__)


class SSHClient:
    def __init__(self, host, user, key_filename, logger=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.logger = logger or logging.getLogger(__name__)
        self.connection = None

    def __enter__(self):
        self.connection = Connection(
            host=self.host,
            user=self.user,
            connect_kwargs={"key_filename": self.key_filename},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    def run_command(self, command):
        result = self.connection.run(command, hide=True)
        return result.stdout.strip()

    def download(self, remote, local, output=None):
        self.connection.get(remote, local=local)
        self.logger.debug(f"{remote} downloaded to {local}")

        if output is not None:
            self.logger.info("Remote command output: %s", output)
        file_size = os.path.getsize(local)
        self.logger.info("Downloaded %s to %s (size: %s)", remote, local, file_size)

    def close(self):
        if self.connection:
            self.connection.close()


if __name__ == "__main__":
    host = "openmediavault"
    user = "root"
    key_filename = "yourkeyhere"
    command = 'echo "Hello from Fabric!"'
    remote_syslog = "/var/log/syslog"
    local_syslog = "temp/syslog"

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    with SSHClient(host, user, key_filename, logger) as ssh:
        try:
            output = ssh.run_command(command)
            print(output)
            ssh.download(remote_syslog, local_syslog, output)
        finally:
            ssh.close()
