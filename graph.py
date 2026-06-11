import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain.messages import AIMessage, HumanMessage
import nodes
import sys
import json
import time
import subprocess

MAX_REFLECTION_LOOPS = 50
MAX_CONTEXT_MESSAGES = 20


def route_tool_response(state: nodes.AgentState) -> Literal["testwriter", "programmer", "planner", "evaluator"]:
    return state["active_node"]


def route_planner(state: nodes.AgentState) -> Literal["tool_node", "programmer", "planner", "compactor", END]:
    log = state["node_messages"].get("planner", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "programmer"
    if state.get("timeout"):
        return "compactor"
    if state.get("finished"):
        return END
    return "planner"


def route_testwriter(state: nodes.AgentState) -> Literal["tool_node", "evaluator", "testwriter", "compactor"]:
    log = state["node_messages"].get("testwriter", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "evaluator"
    if state.get("timeout") or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "testwriter"


def route_programmer(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "testwriter", "compactor"]:
    log = state["node_messages"].get("programmer", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        if state['fault'] == "programmer":
            return "evaluator"
        return "testwriter"
    if state.get("timeout") or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "programmer"

def route_evaluator(state: nodes.AgentState) -> Literal["tool_node", "programmer", "testwriter", "evaluator", "context_cleaner", "compactor"]:
    log = state["node_messages"].get("evaluator", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state.get("passed"):
        return "context_cleaner"
    fault = state.get("fault")
    if fault:
        return fault
    if state.get("timeout") or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "evaluator"

def route_compactor(state: nodes.AgentState) -> Literal["testwriter", "programmer", "planner", "evaluator"]:
    return state["active_node"]

graph = StateGraph(nodes.AgentState)
graph.add_node("planner", nodes.planner_node)
graph.add_node("testwriter", nodes.testwriter_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("programmer", nodes.programmer_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("evaluator", nodes.evaluator_node)
graph.add_node("tool_node", nodes.tool_node)
graph.add_node("context_cleaner", nodes.context_cleaner_node)
graph.add_node("compactor", nodes.compactor_node)

graph.add_edge(START, "planner")
graph.add_edge("context_cleaner", "planner")

graph.add_conditional_edges("planner", route_planner, ["tool_node", "programmer", "planner", "compactor", END])
graph.add_conditional_edges("testwriter", route_testwriter, ["tool_node", "evaluator", "testwriter", "compactor"])
graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "testwriter", "compactor"])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "testwriter", "context_cleaner", "evaluator", "compactor"])
graph.add_conditional_edges("tool_node", route_tool_response, ["testwriter", "programmer", "planner", "evaluator", "compactor"])
graph.add_conditional_edges("compactor", route_compactor, ["testwriter", "programmer", "planner", "evaluator", "compactor"])

agent = graph.compile()

async def runloop(prompt: str,):
    result = await agent.ainvoke({
        "node_messages": {"planner": [HumanMessage(content=prompt)]},
        "reflection_count": 0,
        "finished": False,
        "fault": '',
        "node_completed": False
    })
    return result

if __name__ == "__main__":

    codegraph = subprocess.Popen(
        ["codegraph-mcp", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(3)

        prompt_file = sys.argv[1]

        with open(prompt_file) as f:
            prompt = f.read()

        result = asyncio.run(runloop(prompt))

        with open("/app/workspace/_result.json", "w") as f:
            json.dump({
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            }, f)

    finally:
        codegraph.terminate()

        try:
            codegraph.wait(timeout=10)
        except subprocess.TimeoutExpired:
            codegraph.kill()