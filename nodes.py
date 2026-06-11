import asyncio
import json
import re
from typing import Literal
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from pydantic import ValidationError
from typing_extensions import TypedDict
import tools
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()
minimax_api_key = os.getenv("MINIMAX_API_KEY")


class AgentState(TypedDict):
    node_messages: dict
    reflection_count: int
    passed: bool
    start_time: float
    elapsed_time: float
    input_tokens: int
    output_tokens: int
    fixing: bool
    fault: str
    task: str
    timeout: bool
    finished: bool
    node_completed: bool
    active_node: Literal["testwriter", "programmer", "planner", "evaluator"]


MODEL_NAME = "MiniMax-M3"
BASE_URL = "https://api.minimax.io/v1"

testwriter_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
programmer_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
evaluator_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
planner_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
compactor_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)

cg_tools = tools.codegraph_tools

all_tools = [tools.read_file, tools.bash, tools.write_file, tools.str_replace, *cg_tools]
tools_by_name = {tool.name: tool for tool in all_tools}

worker_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash, *cg_tools]
evaluator_tools = [tools.read_file, tools.bash, *cg_tools]
planner_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash, *cg_tools]

testwriter_model = testwriter_model.bind_tools(worker_tools)
programmer_model = programmer_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)
planner_model = planner_model.bind_tools(planner_tools)

def remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def get_content(response) -> str:
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list) and response.content:
        return response.content[0].get("text", "")
    return ""


def format_messages(state: AgentState, node: str) -> str:
    messages = state.get("node_messages", {}).get(node, [])
    if not messages:
        return ""
    lines = []
    for msg in messages:
        content = get_content(msg)
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {content}")
        elif isinstance(msg, AIMessage):
            if content:
                lines.append(f"{node.upper()} (YOU): {content}")
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                lines.append(f"{node.upper()} (YOU) called tool `{tc['name']}({args})`")
        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL RESULT:\n{content}")
    return "\n\n".join(lines)


async def planner_node(state: AgentState):
    systemprompt = """
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.
"""
    humanmessage = f"""
Your job is to plan the development of a software codebase.
The start.md file are the requirements the system must meet. Read it and
explore the current state of the workspace to concisely divide the code implementation into tasks.
If a task's results are reasonably testable programatically, describe how it should be tested.
Describe the folder and file structure. Write the plan details to store.json (check syntax is valid), which will persist across loops.
Decide the current task based on the state of the workspace, you can modify existing tasks if needed so that the codebase mirrors the requirements.
File structure is critical, the project cannot be finished until every file in the requirements spec exists and is implemented.

store.json expected schema:
{{
    "project_info": dict with any relevant info about the project (e.g. programming language, framework, libraries, etc) that can be useful for implementation and testing,
    "current_task": {{
            "task": "Short task description, function signature if applicable, and any relevant details for implementation.",
            "test_instructions": "Instructions to implement the test for this task, including any relevant comments from the test file."
            "mentioned_files": ["list", "of", "files", "relevant", "to", "this", "task", "for", "implementation"]
    }}
    "upcoming_tasks": [ tasks like current_task ],
    "file_structure": {{
        "folder1": {{
            "subfolder1": ["file1.py", "file2.py"],
            ...
        }},
        ...
    }}
}}

Your expected response schema when store.json is written:
{{
    "status": "PLANNING_DONE|PROJECT_DONE"  #When planning is done but the codebase is not ready yet, reply PLANNING_DONE When the codebase mirrors the start.md requirements, reply PROJECT_DONE
}}

Your log (last is most recent):
{format_messages(state, 'planner')}
"""
    print(f'PLANNER PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            planner_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        
        node_completed = False
        finished = False
        task = ''

        cleaned = remove_think_tags(get_content(response))
        try:
            cleaned = json.loads(cleaned)
        except Exception as e:
            pass
        if isinstance(cleaned, dict):
            print(get_content(response))
            if cleaned.get('status', '') == "PLANNING_DONE":
                plan = json.load(open("workspace/store.json"))
                task = plan.get("current_task", '')
                node_completed = True
            if cleaned.get('status', '') == "PROJECT_DONE":
                finished = True

        return {
            "active_node": "planner",
            "node_messages": {
                **state["node_messages"],
                "planner": state["node_messages"].get("planner", []) + [response]
            },
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "node_completed": node_completed,
            "finished": finished,
            "timeout": False,
            "task": task
        }
    except asyncio.TimeoutError:
        print("PLANNER TIMEOUT")
        return {"passed": False,
                "timeout": True}


async def programmer_node(state: AgentState):
    systemprompt = """
You are an expert software programmer.
"""
    humanmessage = f"""
Your task:
{state.get("task", "No tasks yet.")}

Do the task but DO NOT write the tests for it, the tests will be done by another agent. Focus only on the implementation.
DO NOT run the tests, an evaluator will do it. Make corrections based on feedback from the evaluator.
When it's done, just reply DONE

You can find info about the project at store.json.
You can query code structure with the graph tools.

Your log (last is most recent):
{format_messages(state, 'programmer')}
"""
    print(f'programmer PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            programmer_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        node_completed = "DONE" == remove_think_tags(get_content(response))
        return {
            "active_node": "programmer",
            "node_messages": {
                **state["node_messages"],
                "programmer": state["node_messages"].get("programmer", []) + [response] if not node_completed else []
            },
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("programmer TIMEOUT")
        return {"passed": False, "timeout": True}


async def testwriter_node(state: AgentState):
    systemprompt = """
You are an expert software tester.
"""
    humanmessage = f"""
Your task:
{state.get("task", "No tasks yet.")}

ONLY write tests for the task if they are needed, DO NOT implement the actual code, another agent is doing that.
DO NOT run the tests, an evaluator will do it. Make corrections based on feedback from the evaluator.
When it's done, just reply DONE

You can find info about the project at store.json.
You can query code structure with the graph tools.

Your log (last is most recent):
{format_messages(state, 'testwriter')}
"""
    print(f'testwriter PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            testwriter_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        node_completed = "DONE" == remove_think_tags(get_content(response))
        return {
            "active_node": "testwriter",
            "node_messages": {
                **state["node_messages"],
                "testwriter": state["node_messages"].get("testwriter", []) + [response] if not node_completed else []
            },
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("testwriter TIMEOUT")
        return {"passed": False, "timeout": True}


async def evaluator_node(state: AgentState):
    systemprompt = """
You are an expert QA evaluator.
For vitest, always use 'npx vitest run' or 'npm test -- --run' to avoid watch mode blocking.
"""
    humanmessage = f"""
Current task:
{state.get("task", "No tasks yet.")}

If the current task has tests, run them without cache and check if they pass.
You can use any tools available to run the tests and check results, but DO NOT modify the code or tests, just run them and observe the output.
You can find info about the project at store.json.
You can query code structure with the graph tools.

If the tests pass or they aren't needed, reply with GOOD. If they fail, determine whose fault it is, the PROGRAMMER or the TESTWRITER and
reply with the output. DO NOT attempt to fix, just report.

Example fail response:
TESTWRITER
[stacktrace here]

Your log (last is most recent):
{format_messages(state, 'evaluator')}
"""
    print(f'EVALUATOR PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            evaluator_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        content = remove_think_tags(get_content(response))

        passed = "GOOD" in content
        fault = "testwriter" if "TESTWRITER" in content else "programmer" if "PROGRAMMER" in content else ""
        evaluation_ended = passed or (fault != "")
        
        response_copy = response.model_copy()
        response_copy.content = "The evaluator says: " + content
        return {
            "active_node": "evaluator",
            "node_messages": {
                **state["node_messages"],
                "evaluator": state["node_messages"].get("evaluator", []) + [response] if not evaluation_ended else [],
                fault: state["node_messages"].get(fault, []) + [response_copy],
            },
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "passed": passed,
            "fault": fault,
            "reflection_count": state.get("reflection_count", 0) + 1,
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("EVALUATOR TIMEOUT")
        return {"passed": False, "timeout": True}



async def compactor_node(state: AgentState):
    systemprompt = """
You are a helpful assistant that compacts the conversation history to save tokens, keeping only the most relevant information.
"""
    humanmessage = f"""
Compact this conversation history, keeping only the most relevant information for the current task and removing redundant or irrelevant details.
The conversation belongs to an agent in a software development loop.

The log to compact (last is most recent):
{format_messages(state, state['active_node'])}
"""
    print(f'COMPACTOR PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            compactor_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        active_node = state['active_node']
        return {
            "node_messages": {
                **state["node_messages"],
                active_node: [response]
            },
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("compactor TIMEOUT")
        return {"passed": False, "timeout": True}



async def tool_node(state: AgentState):
    active = state["active_node"]
    last_message = state["node_messages"][active][-1]
    results = []

    for tool_call in last_message.tool_calls:
        try:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
        except ValidationError as e:
            observation = (
                f"INVALID TOOL CALL\n\n"
                f"Tool: {tool_call.get('name')}\n"
                f"Arguments: {tool_call.get('args')}\n\n"
                f"Validation error:\n{e}"
            )
        except Exception as e:
            observation = (
                f"TOOL EXECUTION FAILED\n\n"
                f"Tool: {tool_call.get('name')}\n"
                f"Arguments: {tool_call.get('args')}\n\n"
                f"Error:\n{type(e).__name__}: {e}"
            )
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {
        "node_messages": {
            **state["node_messages"],
            active: state["node_messages"].get(active, []) + results
        }
    }



async def context_cleaner_node(state: AgentState):
    print("*****************CLEANING CONTEXT*******************")
    return {
        "node_messages": {
            k: [] for k in state["node_messages"].keys()
        },
        "passed": False,
        "reflection_count": 0
    }