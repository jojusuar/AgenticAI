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
from memory import MyMemory

MAX_FEEDBACK_LOOPS_PER_TASK = 3
MAX_PLANNER_RETRIES = 5
MAX_CONTEXT_MESSAGES = 30
MAX_INPUT_TOKENS = 8000000
MAX_OUTPUT_TOKENS = 500000

def budget_exceeded(state: nodes.AgentState) -> bool:
    return (
        state.get("input_tokens", 0) >= MAX_INPUT_TOKENS
        or state.get("output_tokens", 0) >= MAX_OUTPUT_TOKENS
    )

def route_tool_response(state: nodes.AgentState) -> Literal["planner", "programmer", "evaluator", "compactor"]:
    return state.get('active_node')

def route_planner(state: nodes.AgentState) -> Literal["tool_node", "programmer", "planner", "compactor", END]:
    if budget_exceeded(state) or state.get("planner_retries", 0) >= MAX_PLANNER_RETRIES:
        return END
    memory = state.get('memory')
    log = memory.l1.get("planner", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        if state.get("finished"):
            return END
        return "programmer"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    return "planner"

def route_programmer(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "compactor", END]:
    if budget_exceeded(state):
        return END
    memory = state.get('memory')
    log = memory.l1.get("programmer", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "evaluator"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    return "programmer"

def route_evaluator(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "compactor", "memory_operator", "planner", END]:
    if budget_exceeded(state):
        return END
    memory = state.get('memory')
    if state.get("reflection_count") >= MAX_FEEDBACK_LOOPS_PER_TASK:
        if getattr(memory, "monolithic", False):
            return "planner"
        return "memory_operator"
    log = memory.l1.get("evaluator", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    if not state.get("node_completed", False):
        return "evaluator"
    if state.get("passed"):
        if getattr(memory, "monolithic", False):
            return "planner"
        return "memory_operator"
    return "programmer"

def route_compactor(state: nodes.AgentState) -> Literal["programmer", "evaluator", "compactor", END]:
    if budget_exceeded(state):
        return END
    return state.get('active_node')

def route_memory_operator(state: nodes.AgentState) -> Literal["planner", END]:
    if state.get('finished'):
        return END
    return "planner"


graph = StateGraph(nodes.AgentState)
graph.add_node("planner", nodes.planner_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("programmer", nodes.programmer_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("evaluator", nodes.evaluator_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("tool_node", nodes.tool_node)
graph.add_node("memory_operator", nodes.memory_operator_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("compactor", nodes.compactor_node, retry_policy=RetryPolicy(max_attempts=1))

graph.add_edge(START, "planner")

graph.add_conditional_edges("planner", route_planner, ["tool_node", "programmer", "planner", "compactor", END])
graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "evaluator", "compactor", "memory_operator", "planner", END])
graph.add_conditional_edges("tool_node", route_tool_response, ["planner", "programmer", "evaluator", "compactor"])
graph.add_conditional_edges("compactor", route_compactor, ["programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("memory_operator", route_memory_operator, ["planner", END])

agent = graph.compile()

def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


async def runloop(prompt: str, memory: bool = True):
    result = await agent.ainvoke({
        "memory": MyMemory(memory=memory),
        "reflection_count": 0,
        "finished": False,
        "node_completed": False,
        "current_task": {},
        "planner_retries": 0
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
        memory = True
        for arg in sys.argv[2:]:
            if arg.startswith("memory="):
                memory = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith("--memory="):
                memory = parse_bool(arg.split("=", 1)[1])

        with open(prompt_file) as f:
            prompt = f.read()

        result = asyncio.run(runloop(prompt, memory=memory))

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


#Pullear todas las imagenes de test del benchmark, porsiaca las tumban
