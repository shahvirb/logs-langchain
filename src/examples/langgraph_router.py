import chainlit as cl
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
import random
import string
import logging
import uuid

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Mock LLM and Tool Functions ---


def mock_llm_router(
    user_input: str,
) -> Literal["joke", "random_number", "random_letter", "unknown"]:
    """Simulates an LLM call to determine user intent."""
    logger.info(f"Mock LLM processing input: '{user_input}'")
    if "joke" in user_input.lower():
        return "joke"
    elif "number" in user_input.lower():
        return "random_number"
    elif "letter" in user_input.lower():
        return "random_letter"
    else:
        logger.warning("LLM couldn't determine a clear intent.")
        return "unknown"


def generate_random_number_tool() -> dict:
    """Tool to generate a random number."""
    number = random.randint(1, 100)
    logger.info(f"Random number tool generated: {number}")
    return {"tool_output": number, "tool_name": "random_number_tool"}


def generate_random_capital_letter_tool() -> dict:
    """Tool to generate a random capital letter."""
    letter = random.choice(string.ascii_uppercase)
    logger.info(f"Random letter tool generated: {letter}")
    return {"tool_output": letter, "tool_name": "random_capital_letter_tool"}


# --- LangGraph State Definition ---


class AgentState(TypedDict):
    user_input: str
    llm_decision: Literal["joke", "random_number", "random_letter", "unknown"] | None
    tool_output: Annotated[str | int | None, lambda x, y: y if y is not None else x] = (
        None  # Accumulate tool output
    )
    final_message: str | None


# --- LangGraph Nodes ---


async def router_node(state: AgentState):
    """Determines the next step based on user input."""
    logger.info("--- Router Node ---")
    user_input = state["user_input"]
    await cl.Message(content=f"🤖 Router deciding for: '{user_input}'").send()
    decision = mock_llm_router(user_input)
    logger.info(f"Router decision: {decision}")
    return {"llm_decision": decision}


async def joke_node(state: AgentState):
    """Handles the 'joke' path."""
    logger.info("--- Joke Node ---")
    joke_message = "Okay, here's a programming joke: Why do programmers prefer dark mode? Because light attracts bugs! 🐞"
    logger.info(f"Joke to be told: {joke_message}")
    await cl.Message(content=joke_message).send()
    return {"final_message": joke_message}


async def tool_node_number(state: AgentState):
    """Calls the random number tool."""
    logger.info("--- Tool Node: Random Number ---")
    await cl.Message(content="⚙️ Generating a random number...").send()
    result = generate_random_number_tool()
    return {"tool_output": result}


async def tool_node_letter(state: AgentState):
    """Calls the random letter tool."""
    logger.info("--- Tool Node: Random Letter ---")
    await cl.Message(content="⚙️ Generating a random capital letter...").send()
    result = generate_random_capital_letter_tool()
    return {"tool_output": result}


async def process_tool_output_node(state: AgentState):
    """Processes and logs the output from a tool."""
    logger.info("--- Process Tool Output Node ---")
    tool_output_data = state.get("tool_output")
    if tool_output_data:
        tool_name = tool_output_data.get("tool_name", "Unknown Tool")
        output_value = tool_output_data.get("tool_output", "No Output")
        processed_message = f"Output from {tool_name}: {output_value}"
        logger.info(f"Processed tool output: {processed_message}")
        await cl.Message(content=f"✅ Tool Result: {processed_message}").send()
        return {"final_message": processed_message}
    else:
        logger.warning("No tool output found to process.")
        await cl.Message(content="⚠️ No tool output to process.").send()
        return {"final_message": "No tool output received."}


async def unknown_intent_node(state: AgentState):
    """Handles cases where the intent is unclear."""
    logger.info("--- Unknown Intent Node ---")
    message = "🤔 I'm not sure what you want to do. Please ask for a 'joke', 'random number', or 'random letter'."
    logger.info(message)
    await cl.Message(content=message).send()
    return {"final_message": message}


# --- LangGraph Graph Definition ---


def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("tell_joke", joke_node)
    workflow.add_node("run_number_tool", tool_node_number)
    workflow.add_node("run_letter_tool", tool_node_letter)
    workflow.add_node("process_tool_output", process_tool_output_node)
    workflow.add_node("handle_unknown", unknown_intent_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        lambda x: x["llm_decision"],
        {
            "joke": "tell_joke",
            "random_number": "run_number_tool",
            "random_letter": "run_letter_tool",
            "unknown": "handle_unknown",
        },
    )

    # Add edges from tool nodes to the processing node
    workflow.add_edge("run_number_tool", "process_tool_output")
    workflow.add_edge("run_letter_tool", "process_tool_output")

    # Add end points
    workflow.add_edge("tell_joke", END)
    workflow.add_edge("process_tool_output", END)
    workflow.add_edge("handle_unknown", END)

    # Compile the graph
    # Using MemorySaver for simplicity in this example.
    # For production, you might use a persistent checkpoint like LangGraphCloud or a database.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logger.info("LangGraph compiled successfully.")
    return app


# --- Chainlit Integration ---


@cl.on_chat_start
async def start_chat():
    graph = build_graph()
    cl.user_session.set("graph", graph)
    cl.user_session.set(
        "thread_id", str(uuid.uuid4())
    )  # Create a unique thread_id for the session
    logger.info("Chainlit chat started. Graph initialized.")
    await cl.Message(
        content="Hi! I can tell you a joke, generate a random number, or a random capital letter. What would you like?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    if not graph:
        await cl.Message(
            content="Error: Graph not initialized. Please restart the chat."
        ).send()
        return

    inputs = {"user_input": message.content}
    # Create a loading message
    loading_msg = cl.Message(content="🧠 Thinking...")
    await loading_msg.send()

    # Stream the graph execution
    # The `astream_events` method allows us to get updates as they happen.
    # We are interested in the 'on_chat_model_stream' for LLM token streaming (not used here directly but good to know)
    # and 'on_tool_end' if we had actual LangChain tools.
    # For this example, we'll primarily see node executions via our logging and cl.Message updates.

    final_state = None
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = event["event"]
        # You can inspect events here if needed for more granular UI updates
        # For example, `if kind == "on_chain_start":` or `if kind == "on_tool_start":`
        # logger.debug(f"Graph Event: {event}")
        if kind == "on_graph_end":  # or on_stream_end for some configurations
            final_state = event["data"]["output"]

    # The main messages are sent from within the nodes themselves.
    # We can send a final confirmation or summary if needed, based on `final_state`.
    if final_state and final_state.get("final_message"):
        # This message is already sent by the last node, so this is just for confirmation if needed.
        # await cl.Message(content=f"Processed: {final_state['final_message']}").send()
        logger.info(
            f"Graph execution finished. Final message: {final_state['final_message']}"
        )
    else:
        logger.info("Graph execution finished, but no specific final message in state.")

    await loading_msg.remove()  # Remove the "Thinking..." message


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
