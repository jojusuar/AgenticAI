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

MAX_REFLECTION_LOOPS = 3
MAX_CONTEXT_MESSAGES = 30
MAX_INPUT_TOKENS = 3000000
MAX_OUTPUT_TOKENS = 200000

def budget_exceeded(state: nodes.AgentState) -> bool:
    return (
        state.get("input_tokens", 0) >= MAX_INPUT_TOKENS
        or state.get("output_tokens", 0) >= MAX_OUTPUT_TOKENS
    )

def route_tool_response(state: nodes.AgentState) -> Literal["programmer", "evaluator", "compactor"]:
    return state["active_node"]


def route_programmer(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "compactor", END]:
    if budget_exceeded(state):
        return END
    log = state["node_messages"].get("programmer", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "evaluator"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    return "programmer"

def route_evaluator(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "compactor", END]:
    if budget_exceeded(state) or state.get("reflection_count") >= MAX_REFLECTION_LOOPS:
        return END
    log = state["node_messages"].get("evaluator", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state.get("passed"):
        return END
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    return "programmer"

def route_compactor(state: nodes.AgentState) -> Literal["programmer", "evaluator", "compactor", END]:
    if budget_exceeded(state):
        return END
    return state["active_node"]

graph = StateGraph(nodes.AgentState)
graph.add_node("programmer", nodes.programmer_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("evaluator", nodes.evaluator_node)
graph.add_node("tool_node", nodes.tool_node)
graph.add_node("compactor", nodes.compactor_node)

graph.add_edge(START, "programmer")

graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("tool_node", route_tool_response, ["programmer", "evaluator", "compactor"])
graph.add_conditional_edges("compactor", route_compactor, ["programmer", "evaluator", "compactor", END])

agent = graph.compile()

async def runloop(prompt: str,):
    result = await agent.ainvoke({
        "node_messages": {"planner": [HumanMessage(content=prompt)]},
        "reflection_count": 0,
        "finished": False,
        "blame": '',
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

        with open("/app/workspace/usage.json", "w") as f:
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


# IDEAS:
# tal vez el approach de cache jerarquico es el mejor?
# con un ponderado entre lru y lfu como politica?
# investigar papers sobre nuevas politicas de manejo de memoria

# idea de titulo: MYARCH: Hierarchical agentic memory for ground-up codebase generation