from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_agent.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    def retrieve(state: AgentState):
        return state

    graph.add_node("retrieve", retrieve)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    return graph.compile()
