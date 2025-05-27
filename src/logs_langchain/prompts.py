from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Optional
from pydantic import BaseModel, Field


class ServerName(BaseModel):
    name: Optional[str] = Field(
        description="The name of the server the user is asking about"
    )


prompt_server_name_identification = ChatPromptTemplate.from_messages(
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
            
            Respect these format instructions for your return value formatting:
            {format_instructions}
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

agent_identification = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an action router for an AI agent system. Your task is to analyze user requests and determine which of the following predefined actions should be performed:

            1.  **read_syslog**: This action involves reading and analyzing system log files (syslog).
            2.  **run_command**: This action involves executing a command and analyzing its output.

            Based on the user's request provided below, identify if it aligns with either the 'read_syslog' or 'run_command' action.

            Your output must be **only** one of the following exact literals:
            * `read_syslog`
            * `run_command`
            * `NONE`

            If the user's request clearly indicates the need to read or analyze system logs, output `read_syslog`.
            If the user's request clearly indicates the need to execute a command or analyze the output of a command, output `run_command`.
            If the user's request does not clearly match either 'read_syslog' or 'run_command', output `NONE`.

            Do not include any other text, explanation, or punctuation in your output besides the exact literal.

            """,
        ),
        ("user", "{question}"),
    ]
)

linux_command_determination = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert Linux sysadmin. You will determine and return the linux shell commands as the next step in our debugging process. You will never say anything other than shell commands to debug. Read the following question carefully now and identify your next steps and be sure to return only the shell commands. You will always assume you are debugging from inside the server, in other words you have already SSH'd inside.

            """,
        ),
        ("user", "Question: {question}"),
    ]
)

expert_linux_debugger = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert Linux debugger. Analyze the user's problem and reference the provided debugging command and its output to deduce the answer to their question. If they refer to a hostname or server name you may always assume you are inside the server. Always begin by showing the command which was run and its output then explain quickly whether this addressed their question.
            """,
        ),
        (
            "user",
            "Question: {question}\nDebug Command: {command}\nOutput of running debug command:\n{output}",
        ),
    ]
)


explain_command_result = HumanMessage(
    """
    You are an expert sysadmin speaking to another sysadmin about the above chat history. You will reply back in this format:
    Show the user the command you ran and its output.
    Then explain the command output in detail, including any supporting evidence if necessary.
    """
)


class DangerousCommand(BaseModel):
    is_dangerous: bool = Field(
        description="True if the command is dangerous, False otherwise."
    )
    reason: Optional[str] = Field(
        default=None,
        description="An optional reason explaining why the command is considered dangerous.",
    )


dangerous_command_verification = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a security-conscious AI review assistant. Your task is to review the user's request and determine if it involves executing a potentially dangerous command that could harm the system or data integrity.
            A command is considered dangerous if it has the potential to:
            - Delete files or directories
            - Writing new files, editing files, touching files.
            - Modify system configurations
            - Change user permissions
            - Stop critical services
            
            Respect these format instructions for your return value formatting:
            {format_instructions}
            """,
        ),
        ("user", "Command: {command}"),
    ]
)
