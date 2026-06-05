import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain.messages import AIMessage, HumanMessage
import nodes

MAX_REFLECTION_LOOPS = 3


def route_tool_response(state: nodes.AgentState) -> Literal["worker1", "worker2", "planner", "test_evaluator"]:
    return state["active_node"]


def planner_tool_call(state: nodes.AgentState) -> Literal["tool_node", "worker1"]:
    last_message = state["node_messages"].get("planner", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "worker1"


def worker1_tool_call(state: nodes.AgentState) -> Literal["tool_node", "worker2"]:
    last_message = state["node_messages"].get("worker1", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "worker2"


def worker2_tool_call(state: nodes.AgentState) -> Literal["tool_node", "test_evaluator"]:
    last_message = state["node_messages"].get("worker2", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return "test_evaluator"


def evaluator_tool_call(state: nodes.AgentState) -> Literal["tool_node", "worker2", END]:
    last_message = state["node_messages"].get("test_evaluator", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state["passed"] or state.get("reflection_count", 0) >= MAX_REFLECTION_LOOPS:
        return END
    return "worker2"


graph = StateGraph(nodes.AgentState)
graph.add_node("planner", nodes.planner_node)
graph.add_node("worker1", nodes.worker1_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("worker2", nodes.worker2_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("test_evaluator", nodes.test_evaluator_node)
graph.add_node("tool_node", nodes.tool_node)

graph.add_edge(START, "planner")

graph.add_conditional_edges("planner", planner_tool_call, ["tool_node", "worker1"])
graph.add_conditional_edges("worker1", worker1_tool_call, ["tool_node", "worker2"])
graph.add_conditional_edges("worker2", worker2_tool_call, ["tool_node", "test_evaluator"])
graph.add_conditional_edges("test_evaluator", evaluator_tool_call, ["tool_node", "worker2", END])
graph.add_conditional_edges("tool_node", route_tool_response, ["worker1", "worker2", "planner", "test_evaluator"])
agent = graph.compile()


async def runloop(prompt: str):
    with open("graph.png", "wb") as f:
        f.write(agent.get_graph(xray=True).draw_mermaid_png())
    result = await agent.ainvoke({
        "node_messages": {"planner": [HumanMessage(content=prompt)]},
        "reflection_count": 0,
        "passed": False,
    })

prompt = f'''
Read the REQUIREMENTS.md file. Plan the code implementation in self-contained tasks 
that can be performed in parallel. Describe the directory and file structure.
Each task must have its own test. A final integration test is also required.
'''

asyncio.run(runloop(prompt))