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
MAX_CONTEXT_MESSAGES = 20
MAX_INPUT_TOKENS = 5000000
MAX_OUTPUT_TOKENS = 500000

def budget_exceeded(state: nodes.AgentState) -> bool:
    return (
        state.get("input_tokens", 0) >= MAX_INPUT_TOKENS
        or state.get("output_tokens", 0) >= MAX_OUTPUT_TOKENS
    )

def route_tool_response(state: nodes.AgentState) -> Literal["entrypoint", "testwriter", "programmer", "planner", "evaluator"]:
    return state["active_node"]


def route_entrypoint(state: nodes.AgentState) -> Literal["entrypoint", "planner", "tool_node"]:
    log = state["node_messages"].get("entrypoint", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "planner"
    return "entrypoint"


def route_planner(state: nodes.AgentState) -> Literal["tool_node", "programmer", "planner", "compactor", END]:
    if budget_exceeded(state):
        return END
    log = state["node_messages"].get("planner", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "programmer"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    if state.get("finished"):
        return END
    return "planner"


def route_testwriter(state: nodes.AgentState) -> Literal["tool_node", "evaluator", "testwriter", "compactor", END]:
    if budget_exceeded(state):
        return END
    log = state["node_messages"].get("testwriter", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "evaluator"
    if state.get("timeout") or state.get('model_error') or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "testwriter"


def route_programmer(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "testwriter", "compactor", END]:
    if budget_exceeded(state):
        return END
    log = state["node_messages"].get("programmer", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        if state['blame'] == "programmer":
            return "evaluator"
        return "testwriter"
    if state.get("timeout") or state.get('model_error') or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "programmer"

def route_evaluator(state: nodes.AgentState) -> Literal["tool_node", "programmer", "testwriter", "evaluator", "context_cleaner", "compactor", END]:
    if budget_exceeded(state):
        return END
    log = state["node_messages"].get("evaluator", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state.get("passed") or state.get("reflection_count") >= MAX_REFLECTION_LOOPS:
        return "context_cleaner"
    blame = state.get("blame")
    if blame:
        return blame
    if state.get("timeout") or state.get('model_error') or len(log) >= MAX_CONTEXT_MESSAGES:
        return "compactor"
    return "evaluator"

def route_compactor(state: nodes.AgentState) -> Literal["testwriter", "programmer", "planner", "evaluator", END]:
    if budget_exceeded(state):
        return END
    return state["active_node"]

graph = StateGraph(nodes.AgentState)
graph.add_node('entrypoint', nodes.entrypoint_node)
graph.add_node("planner", nodes.planner_node)
graph.add_node("testwriter", nodes.testwriter_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("programmer", nodes.programmer_node, retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("evaluator", nodes.evaluator_node)
graph.add_node("tool_node", nodes.tool_node)
graph.add_node("context_cleaner", nodes.context_cleaner_node)
graph.add_node("compactor", nodes.compactor_node)

graph.add_edge(START, "entrypoint")
graph.add_edge("context_cleaner", "planner")

graph.add_conditional_edges("entrypoint", route_entrypoint, ["entrypoint", "planner", "tool_node"])
graph.add_conditional_edges("planner", route_planner, ["tool_node", "programmer", "planner", "compactor", END])
graph.add_conditional_edges("testwriter", route_testwriter, ["tool_node", "evaluator", "testwriter", "compactor", END])
graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "testwriter", "compactor", END])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "testwriter", "context_cleaner", "evaluator", "compactor", END])
graph.add_conditional_edges("tool_node", route_tool_response, ["entrypoint", "testwriter", "programmer", "planner", "evaluator", "compactor"])
graph.add_conditional_edges("compactor", route_compactor, ["testwriter", "programmer", "planner", "evaluator", "compactor", END])

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
# usar esta arch con store.json como baseline agentico (solo usar el modelo + tool calls podria ser visto como ventaja trivial para nuestra propuesta)
# implementar otra arch sin persistencia en el workspace, sino con memorias en db vectorial / knowledge graph / priority queues etc para componer el contexto
# comparar https://llm-stats.com/benchmarks/nl2repo estos modelos solitos vs arch con store.json vs arch con memoria real
# eso deberia resaltar la ganancia causada por la memoria

# tal vez el approach de cache jerarquico es el mejor?
# con un ponderado entre lru y lfu como politica?
# investigar papers sobre nuevas politicas de manejo de memoria

# idea de titulo: MYARCH: Hierarchical agentic memory for ground-up codebase generation