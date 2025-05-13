import os
import logging
from logs_langchain import factory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LOG_FILE = "/var/log/syslog"  # Path to the log file
LINES_TO_FETCH = 100


def tail(filename, n):
    logging.debug(f"Fetching last {n} lines from {filename}")
    with open(filename, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        lines = []
        line = b""
        pos = end
        while pos > 0 and len(lines) < n:
            pos -= 1
            f.seek(pos)
            char = f.read(1)
            if char == b"\n":
                if line:
                    lines.append(line[::-1].decode())
                    line = b""
            else:
                line += char
        if line:
            lines.append(line[::-1].decode())
        logging.debug(f"Fetched {len(lines)} lines from {filename}")
        return "\n".join(lines[::-1])


if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        logging.error(f"Log file does not exist: {LOG_FILE}")
        raise FileNotFoundError(f"Log file does not exist: {LOG_FILE}")

    file_contents = tail(LOG_FILE, LINES_TO_FETCH)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You going to analyze the log file. Respond concisely."),
            ("user", "Tell me about this log file: {file_contents}"),
        ]
    )
    llm = factory.llm_factory()
    chain = prompt | llm | StrOutputParser()

    try:
        logging.debug("Invoking LLM chain for log analysis.")
        response = chain.invoke({"file_contents": file_contents})
        print(response)
        logging.debug("LLM chain invocation successful.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"\nAn error occurred: {e}")
        print("Please check your API key and ensure it's valid.")
