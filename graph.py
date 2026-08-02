import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain_core.messages import AIMessage, HumanMessage
import nodes
import sys
import json
import time
from functools import wraps
from tools import WORKSPACE
from memory import MyMemory, OllamaEmbeddingError

MAX_FEEDBACK_LOOPS_PER_TASK = 3
MAX_PLANNER_RETRIES = 10
L3_SIMILARITY_THRESHOLD = 0.33
INPUT_RATE_PER_MILLION = 0.30
CACHED_INPUT_RATE_PER_MILLION = 0.30
OUTPUT_RATE_PER_MILLION = 1.20
MONEY_LIMIT_DOLLARS = 0.75
NODE_STALL_TIMEOUT_SECONDS = 15 * 60

def budget_exceeded(state: nodes.AgentState) -> bool:
    return nodes.state_usage_cost(state) >= state.get("money_limit", 0.0)

def node_stall_timeout_exceeded(state: nodes.AgentState) -> bool:
    last_completion = state.get("last_node_completion_time")
    if last_completion is None:
        return False
    return time.monotonic() - last_completion >= NODE_STALL_TIMEOUT_SECONDS

def should_end(state: nodes.AgentState) -> bool:
    return (
        state.get("rate_limited", False)
        or state.get("authentication_failed", False)
        or budget_exceeded(state)
        or node_stall_timeout_exceeded(state)
    )


def determine_termination_reason(
    state: nodes.AgentState,
    run_error: BaseException | None = None,
) -> str:
    if isinstance(run_error, OllamaEmbeddingError):
        return "embedding_failure"
    if isinstance(run_error, nodes.UsageReportingError):
        return "usage_reporting_failure"
    if run_error is not None:
        return "error"
    if state.get("rate_limited", False):
        return "rate_limited"
    if state.get("authentication_failed", False):
        return "authentication_failed"
    if budget_exceeded(state):
        return "money_limit_reached"
    if node_stall_timeout_exceeded(state):
        return "node_stall_timeout"
    if state.get("planner_retries", 0) >= MAX_PLANNER_RETRIES:
        return "planner_retries_exhausted"
    if state.get("finished", False):
        return "finished"
    return "unknown"

def route_tool_response(state: nodes.AgentState) -> Literal["planner", "programmer", "evaluator", "compactor", END]:
    if should_end(state):
        return END
    return state.get('active_node')

def route_planner(state: nodes.AgentState) -> Literal["tool_node", "programmer", "planner", "compactor", END]:
    if should_end(state) or state.get("planner_retries", 0) >= MAX_PLANNER_RETRIES:
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

def route_programmer(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "l2_operator", "compactor", END]:
    if should_end(state):
        return END
    memory = state.get('memory')
    log = memory.l1.get("programmer", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        if memory.l2_enabled:
            return "l2_operator"
        return "evaluator"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    return "programmer"

def route_evaluator(state: nodes.AgentState) -> Literal["tool_node", "programmer", "evaluator", "compactor", "l3_operator", "task_cleanup", "planner", END]:
    if should_end(state):
        return END
    memory = state.get('memory')
    if state.get("reflection_count") >= MAX_FEEDBACK_LOOPS_PER_TASK:
        return "task_cleanup"
    log = memory.l1.get("evaluator", [])
    last_message = log[-1] if log else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state.get("timeout") or state.get('model_error'):
        return "compactor"
    if not state.get("node_completed", False):
        return "evaluator"
    if state.get("passed"):
        if memory.l3_enabled:
            return "l3_operator"
        if getattr(memory, "monolithic", False):
            return "planner"
        return "task_cleanup"
    return "programmer"

def route_compactor(state: nodes.AgentState) -> Literal["planner", "programmer", "evaluator", "compactor", END]:
    if should_end(state):
        return END
    return state.get('active_node')

def route_l3_operator(state: nodes.AgentState) -> Literal["planner", END]:
    if should_end(state) or state.get('finished'):
        return END
    return "planner"


def route_l2_operator(state: nodes.AgentState) -> Literal["evaluator", END]:
    if should_end(state):
        return END
    return "evaluator"


def with_node_heartbeat(node):
    """Refresh graph progress after a normal node completion."""
    @wraps(node)
    async def wrapped(state):
        update = await node(state)
        update = dict(update or {})
        failed = (
            update.get("timeout", False)
            or update.get("model_error", False)
            or update.get("rate_limited", False)
            or update.get("authentication_failed", False)
        )
        if not failed:
            update["last_node_completion_time"] = time.monotonic()
            update["node_stall_timeout"] = False
        return update

    return wrapped


graph = StateGraph(nodes.AgentState)
graph.add_node("planner", with_node_heartbeat(nodes.planner_node), retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("programmer", with_node_heartbeat(nodes.programmer_node), retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("evaluator", with_node_heartbeat(nodes.evaluator_node), retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("tool_node", with_node_heartbeat(nodes.tool_node))
graph.add_node("l2_operator", with_node_heartbeat(nodes.l2_operator_node), retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("l3_operator", with_node_heartbeat(nodes.l3_operator_node), retry_policy=RetryPolicy(max_attempts=1))
graph.add_node("task_cleanup", with_node_heartbeat(nodes.task_cleanup_node))
# Compaction is a failure-recovery action and intentionally does not refresh
# the heartbeat; repeated timeout -> compaction loops must eventually stall out.
graph.add_node("compactor", nodes.compactor_node, retry_policy=RetryPolicy(max_attempts=1))

graph.add_edge(START, "planner")

graph.add_conditional_edges("planner", route_planner, ["tool_node", "programmer", "planner", "compactor", END])
graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "l2_operator", "compactor", END])
graph.add_conditional_edges("l2_operator", route_l2_operator, ["evaluator", END])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "evaluator", "compactor", "l3_operator", "task_cleanup", "planner", END])
graph.add_edge("task_cleanup", "planner")
graph.add_conditional_edges("tool_node", route_tool_response, ["planner", "programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("compactor", route_compactor, ["planner", "programmer", "evaluator", "compactor", END])
graph.add_conditional_edges("l3_operator", route_l3_operator, ["planner", END])

agent = graph.compile()

def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int(value: str, minimum: int = 0) -> int:
    parsed = int(value.strip())
    if parsed < minimum:
        raise ValueError(f"Expected integer >= {minimum}, got {parsed}")
    return parsed


def parse_float(value: str, minimum: float = 0.0, inclusive: bool = True) -> float:
    parsed = float(value.strip())
    valid = parsed >= minimum if inclusive else parsed > minimum
    if not valid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"Expected number {operator} {minimum}, got {parsed}")
    return parsed


def resolve_memory_levels(
    memory: bool | None,
    l1_enabled: bool | None,
    l2_enabled: bool | None,
    l3_enabled: bool | None,
) -> tuple[bool, bool, bool]:
    default = True if memory is None else memory
    return (
        default if l1_enabled is None else l1_enabled,
        default if l2_enabled is None else l2_enabled,
        default if l3_enabled is None else l3_enabled,
    )


async def runloop(
    prompt: str,
    memory: bool | None = None,
    l1_enabled: bool | None = None,
    l2_enabled: bool | None = None,
    l3_enabled: bool | None = None,
    l3_similarity_threshold: float = L3_SIMILARITY_THRESHOLD,
    input_rate_per_million: float = INPUT_RATE_PER_MILLION,
    cached_input_rate_per_million: float = CACHED_INPUT_RATE_PER_MILLION,
    output_rate_per_million: float = OUTPUT_RATE_PER_MILLION,
    money_limit: float = MONEY_LIMIT_DOLLARS,
):
    l1_enabled, l2_enabled, l3_enabled = resolve_memory_levels(
        memory, l1_enabled, l2_enabled, l3_enabled
    )
    graph_memory = MyMemory(
        l1_enabled=l1_enabled,
        l2_enabled=l2_enabled,
        l3_enabled=l3_enabled,
        l3_similarity_threshold=l3_similarity_threshold,
    )
    start_time = time.monotonic()
    if l3_enabled:
        embedding = graph_memory.embed_text("AUTOSOURCE memory embedding preflight")
        print(f"OLLAMA EMBEDDING PREFLIGHT OK ({len(embedding)} dimensions)")

    result = await agent.ainvoke({
        "memory": graph_memory,
        "reflection_count": 0,
        "finished": False,
        "node_completed": False,
        "current_task": {},
        "planner_handoff": {},
        "planner_retries": 0,
        "tasks_attempted": 0,
        "tasks_completed": 0,
        "start_time": start_time,
        "last_node_completion_time": start_time,
        "node_stall_timeout": False,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "harness_input_tokens": 0,
        "harness_cached_input_tokens": 0,
        "harness_output_tokens": 0,
        "harness_reasoning_tokens": 0,
        "memory_input_tokens": 0,
        "memory_cached_input_tokens": 0,
        "memory_output_tokens": 0,
        "memory_reasoning_tokens": 0,
        "input_rate_per_million": input_rate_per_million,
        "cached_input_rate_per_million": cached_input_rate_per_million,
        "output_rate_per_million": output_rate_per_million,
        "money_limit": money_limit,
        "rate_limited": False,
        "authentication_failed": False,
        "authentication_error": ""
    })
    return result

if __name__ == "__main__":
    run_start_time = time.monotonic()
    result = {}
    run_error = None
    memory = None
    l1_enabled = None
    l2_enabled = None
    l3_enabled = None
    resolved_l1, resolved_l2, resolved_l3 = resolve_memory_levels(
        memory, l1_enabled, l2_enabled, l3_enabled
    )
    l3_similarity_threshold = L3_SIMILARITY_THRESHOLD
    input_rate_per_million = INPUT_RATE_PER_MILLION
    cached_input_rate_per_million = CACHED_INPUT_RATE_PER_MILLION
    output_rate_per_million = OUTPUT_RATE_PER_MILLION
    money_limit = MONEY_LIMIT_DOLLARS

    try:
        prompt_file = sys.argv[1]
        for arg in sys.argv[2:]:
            if arg.startswith("memory="):
                memory = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith("--memory="):
                memory = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith(("l1=", "--l1=")):
                l1_enabled = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith(("l2=", "--l2=")):
                l2_enabled = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith(("l3=", "--l3=")):
                l3_enabled = parse_bool(arg.split("=", 1)[1])
            elif arg.startswith(("l3_similarity_threshold=", "--l3-similarity-threshold=")):
                l3_similarity_threshold = parse_float(arg.split("=", 1)[1])
                if l3_similarity_threshold > 1.0:
                    raise ValueError("L3 similarity threshold must be <= 1.0")
            elif arg.startswith(("input_rate=", "--input-rate=")):
                input_rate_per_million = parse_float(arg.split("=", 1)[1])
            elif arg.startswith(("cached_input_rate=", "--cached-input-rate=")):
                cached_input_rate_per_million = parse_float(arg.split("=", 1)[1])
            elif arg.startswith(("output_rate=", "--output-rate=")):
                output_rate_per_million = parse_float(arg.split("=", 1)[1])
            elif arg.startswith(("money_limit=", "--money-limit=")):
                money_limit = parse_float(arg.split("=", 1)[1], inclusive=False)

        with open(prompt_file) as f:
            prompt = f.read()

        resolved_l1, resolved_l2, resolved_l3 = resolve_memory_levels(
            memory, l1_enabled, l2_enabled, l3_enabled
        )
        result = asyncio.run(runloop(
            prompt,
            l1_enabled=resolved_l1,
            l2_enabled=resolved_l2,
            l3_enabled=resolved_l3,
            l3_similarity_threshold=l3_similarity_threshold,
            input_rate_per_million=input_rate_per_million,
            cached_input_rate_per_million=cached_input_rate_per_million,
            output_rate_per_million=output_rate_per_million,
            money_limit=money_limit,
        ))
    except BaseException as e:
        run_error = e
        raise
    finally:
        money_spent = nodes.calculate_usage_cost(
            input_tokens=result.get("input_tokens", 0),
            cached_input_tokens=result.get("cached_input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            input_rate_per_million=input_rate_per_million,
            cached_input_rate_per_million=cached_input_rate_per_million,
            output_rate_per_million=output_rate_per_million,
        )
        harness_money_spent = nodes.calculate_usage_cost(
            input_tokens=result.get("harness_input_tokens", 0),
            cached_input_tokens=result.get("harness_cached_input_tokens", 0),
            output_tokens=result.get("harness_output_tokens", 0),
            input_rate_per_million=input_rate_per_million,
            cached_input_rate_per_million=cached_input_rate_per_million,
            output_rate_per_million=output_rate_per_million,
        )
        memory_money_spent = nodes.calculate_usage_cost(
            input_tokens=result.get("memory_input_tokens", 0),
            cached_input_tokens=result.get("memory_cached_input_tokens", 0),
            output_tokens=result.get("memory_output_tokens", 0),
            input_rate_per_million=input_rate_per_million,
            cached_input_rate_per_million=cached_input_rate_per_million,
            output_rate_per_million=output_rate_per_million,
        )
        termination_reason = determine_termination_reason(result, run_error)
        node_stall_timeout = termination_reason == "node_stall_timeout"
        usage = {
            "input_tokens": result.get("input_tokens", 0),
            "cached_input_tokens": result.get("cached_input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "reasoning_tokens": result.get("reasoning_tokens", 0),
            "harness_usage": {
                "input_tokens": result.get("harness_input_tokens", 0),
                "cached_input_tokens": result.get("harness_cached_input_tokens", 0),
                "output_tokens": result.get("harness_output_tokens", 0),
                "reasoning_tokens": result.get("harness_reasoning_tokens", 0),
                "money_spent": harness_money_spent,
            },
            "memory_overhead": {
                "input_tokens": result.get("memory_input_tokens", 0),
                "cached_input_tokens": result.get("memory_cached_input_tokens", 0),
                "output_tokens": result.get("memory_output_tokens", 0),
                "reasoning_tokens": result.get("memory_reasoning_tokens", 0),
                "money_spent": memory_money_spent,
            },
            "input_rate_per_million": input_rate_per_million,
            "cached_input_rate_per_million": cached_input_rate_per_million,
            "output_rate_per_million": output_rate_per_million,
            "money_limit": money_limit,
            "money_spent": money_spent,
            "money_limit_reached": money_spent >= money_limit,
            "rate_limited": result.get("rate_limited", False),
            "authentication_failed": result.get("authentication_failed", False),
            "authentication_error": result.get("authentication_error", ""),
            "elapsed_seconds": time.monotonic() - run_start_time,
            "finished": result.get("finished", False),
            "tasks_attempted": result.get("tasks_attempted", 0),
            "tasks_completed": result.get("tasks_completed", 0),
            "node_stall_timeout": node_stall_timeout,
            "termination_reason": termination_reason,
            "embedding_failed": termination_reason == "embedding_failure",
            "usage_reporting_failed": (
                termination_reason == "usage_reporting_failure"
            ),
            "memory_enabled": resolved_l1 or resolved_l2 or resolved_l3,
            "l1_enabled": resolved_l1,
            "l2_enabled": resolved_l2,
            "l3_enabled": resolved_l3,
            "l3_similarity_threshold": l3_similarity_threshold,
            "error_type": type(run_error).__name__ if run_error else None,
            "error": str(run_error) if run_error else None,
        }
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        with open(WORKSPACE / "usage.json", "w") as f:
            json.dump(usage, f, indent=2)


#Pullear todas las imagenes de test del benchmark, porsiaca las tumban
