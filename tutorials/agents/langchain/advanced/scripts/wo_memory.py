from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain.chat_models import init_chat_model

import dotenv
import uuid
dotenv.load_dotenv()


llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile()
# graph = graph_builder.compile(checkpointer=memory)

png_graph = graph.get_graph().draw_mermaid_png()
with open("langgraph_memory.png", "wb") as f:
    f.write(png_graph)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        # The graph.stream() method returns an iterator. We can fully consume
        # it to get all events in a list.
        events = list(graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            # config,
            stream_mode="values",
        ))
        
        # The last event in the stream is the final state of the graph
        final_state = events[-1]
        
        # We can access the messages from the final state
        # and print them to see the full conversation history
        print("---- Full Conversation History ----")
        for message in final_state.get("messages", []):
            message.pretty_print()
        print("------------------------------------")