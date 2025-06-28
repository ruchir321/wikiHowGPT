import os
import dotenv

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.state import StateGraph
from langgraph.constants import START, END
from langgraph.graph.message import add_messages

from langchain.chat_models.base import init_chat_model


dotenv.load_dotenv()

# eval
os.environ["LANGSMITH_API_KEY"] = dotenv.dotenv_values()["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = dotenv.dotenv_values()["LANGSMITH_TRACING"]
os.environ["LANGSMITH_PROJECT"] = dotenv.dotenv_values()["LANGSMITH_PROJECT"]


class State(TypedDict):
    # Messages have type "list".
    # The `add_message` function in the annotations defines how the state key should be updated
    # (in this case, it adds messages to the list instead of overwriting them)

    messages: Annotated[list, add_messages]
    # TODO[curiosity]: learn about Reducer pattern in state management

# chatbot application is a state machine. It has states, nodes and edges
graph_builder = StateGraph(State)

model = init_chat_model(
    model="gemma3n:e2b",
    model_provider="ollama"
)

def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

# Add a chatbot node. Node is a python function that takes action to change the graph state
graph_builder.add_node("chatbot", chatbot)

# Add an entry point to start work
graph_builder.add_edge(START, "chatbot")

# Add an exit point
graph_builder.add_edge("chatbot", END)

# Compile the graph to use it as a runnable (invoke, stream, run async.)
graph = graph_builder.compile()

# visualize

graph.get_graph().draw_mermaid_png(output_file_path="./assets/graphs/chatbot_graph_0.png")

from langchain_core.tracers import langchain as langchain_tracer