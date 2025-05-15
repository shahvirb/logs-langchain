from fabric import Connection


def run_remote_command(connection, command):
    result = connection.run(command, hide=True)
    return result.stdout.strip()


if __name__ == "__main__":
    host = "openmediavault"
    user = "root"
    command = 'echo "Hello from Fabric!"'
    remote_syslog = "/var/log/syslog"
    local_syslog = "temp/syslog"

    with Connection(host=host, user=user, connect_kwargs={"key_filename": "yourkeyhere"}) as c:
        output = run_remote_command(c, command)
        print(output)

        c.get(remote_syslog, local=local_syslog)
        print(f"---- /var/log/syslog downloaded to {local_syslog} ----")
