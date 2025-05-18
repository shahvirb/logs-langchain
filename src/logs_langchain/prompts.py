from langchain_core.prompts import ChatPromptTemplate

server_name_identification = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            As a case-sensitive data extraction system, identify and return the server name from the following input. The server name must be extracted with its exact original capitalization. If the server name is not found with its precise casing, output the literal string 'NONE'.
            Input: 'What is happening in server helium?'
            Output: 'helium'
            Input: 'Deploy to Server PROD.'
            Output: 'PROD'
            Input: 'Status on server dev.'
            Output: 'dev'
            Input: 'No server specified in the request.'
            Output: 'NONE'
            """,
        ),
        ("user", "Your turn: {question}"),
    ]
)

sysadmin_log_context_answer = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful sysadmin. Use the log file contents provided to answer the user's question. Quote log lines that support your answer. If the log file does not contain relevant information, respond with 'I don't know'.",
        ),
        ("user", "User's Question: {question}\n\nLogs:\n{logs}"),
    ]
)
