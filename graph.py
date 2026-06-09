import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langchain.messages import AIMessage, HumanMessage
import nodes

MAX_REFLECTION_LOOPS = 50
MAX_CONTEXT_MESSAGES = 20


def route_tool_response(state: nodes.AgentState) -> Literal["testwriter", "programmer", "planner", "evaluator"]:
    return state["active_node"]


def route_planner(state: nodes.AgentState) -> Literal["tool_node", "programmer", "planner", "compactor"]:
    last_message = state["node_messages"].get("planner", [])[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    if state['node_completed']:
        return "programmer"
    if state.get("timeout"):
        return "compactor"
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


def route_context_cleaner(state: nodes.AgentState) -> Literal["programmer", END]:
    if state['finished']:
        return END
    return 'programmer'


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

graph.add_conditional_edges("planner", route_planner, ["tool_node", "programmer", "planner", "compactor"])
graph.add_conditional_edges("testwriter", route_testwriter, ["tool_node", "evaluator", "testwriter", "compactor"])
graph.add_conditional_edges("programmer", route_programmer, ["tool_node", "programmer", "evaluator", "testwriter", "compactor"])
graph.add_conditional_edges("evaluator", route_evaluator, ["tool_node", "programmer", "testwriter", "context_cleaner", "evaluator", "compactor"])
graph.add_conditional_edges("tool_node", route_tool_response, ["testwriter", "programmer", "planner", "evaluator", "compactor"])
graph.add_conditional_edges("compactor", route_compactor, ["testwriter", "programmer", "planner", "evaluator", "compactor"])
graph.add_conditional_edges("context_cleaner", route_context_cleaner, ["programmer", END])

agent = graph.compile()


async def runloop(prompt: str):
    with open("graph.png", "wb") as f:
        f.write(agent.get_graph(xray=True).draw_mermaid_png())
    result = await agent.ainvoke({
        "node_messages": {"planner": [HumanMessage(content=prompt)]},
        "reflection_count": 0,
        "finished": False,
        "fault": '',
        "node_completed": False
    })
    return result

prompt = f'''
Read the REQUIREMENTS.md file. Concisely divide the code implementation into tasks.
If a task's results are reasonably testable programatically, describe how it should be tested.
Describe the folder and file structure. Write the plan details to store.json.
When you are done, just reply PLANNING DONE.

store.json expected schema:
{{
    "project_info": dict with any relevant info about the project (e.g. programming language, framework, libraries, etc) that can be useful for implementation and testing,
    "tasks": [
        {{
            "task": "Short task description, function signature if applicable, and any relevant details for implementation.",
            "test_instructions": "Instructions to implement the test for this task, including any relevant comments from the test file."
            "mentioned_files": ["list", "of", "files", "relevant", "to", "this", "task", "for", "implementation"]
        }},
        ...
    ],
    "file_structure": {{
        "folder1": {{
            "subfolder1": ["file1.py", "file2.py"],
            ...
        }},
        ...
    }}
}}
'''

output = asyncio.run(runloop(prompt))
print(f"Input tokens:  {output['input_tokens']}")
print(f"Output tokens: {output['output_tokens']}")
print(f"Total tokens:  {output['input_tokens'] + output['output_tokens']}")


#TODO: revisar si el evaluator en serio esta respondiendo siempre lo mismo o usando stale tests (pytest_cache?)