import asyncio
import json
import re
from typing import Literal
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage
)
from pydantic import ValidationError
from typing_extensions import TypedDict
import tools
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from memory import MyMemory, remove_think_from_message


load_dotenv()
minimax_api_key = os.getenv("MINIMAX_API_KEY")


class AgentState(TypedDict):
    memory: MyMemory
    project_info: str
    tasks: list
    reflection_count: int
    passed: bool
    start_time: float
    elapsed_time: float
    input_tokens: int
    output_tokens: int
    timeout: bool
    model_error: bool
    finished: bool
    node_completed: bool
    active_node: Literal["programmer", "planner", "evaluator"]


MODEL_NAME = "MiniMax-M3"
BASE_URL = "https://api.minimax.io/v1"

planner_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
programmer_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
evaluator_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
compactor_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
memory_operator_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)

cg_tools = tools.codegraph_tools

all_tools = [tools.read_file, tools.bash, tools.write_file, tools.str_replace]
tools_by_name = {tool.name: tool for tool in all_tools}

worker_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]
evaluator_tools = [tools.read_file, tools.bash, *cg_tools]
planner_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]

programmer_model = programmer_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)
planner_model = planner_model.bind_tools(planner_tools)


async def planner_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'planner'

    systemprompt = """
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.
"""
    humanmessage = f"""
Create a codebase development plan following the specification at start.md

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "project_info": "Project abstract, details about language, dependencies needed, frameworks, etc"
    "tasks": [
        {{
            "task": "Short task description, class definitions or functions signatures if applicable, and any relevant details for implementation and test writing.",
            "test_instructions": "Instructions for the evaluator to test the implementation, like running tests, verifying imports or empirical evaluation, but NO WRITING CODE."
            "relevant_files": ["names", "of", "files", "relevant", "to", "this", "task", "for", "implementation"],
            "target_file_structure": {{
                                        "folder1": {{
                                            "subfolder1": ["file1.py", "file2.py"],
                                            ...
                                        }},
                                        ...
                                    }} ## expected full workspace structure at the end of this particular task
        }},
        {{task2}},
        ...,
        {{task n}},
    ]
}}

{memory.inject(node_name)}
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
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)
        
        node_completed = False

        cleaned = None
        try:
            cleaned = json.loads(cleaned_response.content)
        except Exception as e:
            pass

        task_list = []
        project_info = ''
        if isinstance(cleaned, dict):
            task_list = cleaned.get('tasks', [])
            project_info = cleaned.get('project_info', '')
            if task_list and project_info:
                node_completed = True
            else:
                memory.add_self_message(HumanMessage(content='The response JSON does not follow the schema.'), node_name)

        return {
            "active_node":node_name,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "node_completed": node_completed,
            "timeout": False,
            "tasks": task_list,
            "project_info": project_info
        }
    
    except asyncio.TimeoutError:
        print("PLANNER TIMEOUT")
        return {"passed": False, "node_completed": False, "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"timeout": False, "node_completed": False, "model_error": True}
    

async def programmer_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'programmer'
    info = state.get('project_info', '')
    tasks_list = state.get('tasks', [])
    current_task = tasks_list[0] if tasks_list else {}
    # instructions = current_task.get('task', 'No task available.')
    # evaluator_instructions = current_task.get('test', 'No task available.')
    # relevant_files = current_task.get('relevant_files', [])
    # target_file_structure = current_task.get('target_file_structure', '')

    systemprompt = """
You are an expert software programmer.
"""
    humanmessage = f"""

The current task:
{current_task}

General project info:
{info}

If needed, full spec is at start.md
You must fix what the evaluator tells you to if it gives feedback.
    
Reply with {{"status": "DONE"}} when you are done, JUST THE JSON, no additional text or markdown fences.

{memory.inject(node_name)}
"""
    print(f'{node_name} PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            programmer_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )

        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        node_completed = False
        try:
            status = json.loads(cleaned_response.content)
            node_completed = status['status'] == 'DONE'
        except Exception:
            pass
        
        return {
            "active_node": node_name,
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }

    except asyncio.TimeoutError:
        print(f"{node_name} TIMEOUT")
        return {"passed": False, "node_completed": False, "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"timeout": False, "node_completed": False, "model_error": True}


async def evaluator_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'evaluator'
    tasks_list = state.get('tasks', [])
    info = state.get('project_info', '')
    current_task = tasks_list[0] if tasks_list else {}
    # instructions = current_task.get('test_instructions', 'No task available.')
    # relevant_files = current_task.get('relevant_files', [])
    # target_file_structure = current_task.get('target_file_structure', '')

    systemprompt = """
You are an expert QA evaluator.
"""
    humanmessage = f"""
The current task:
{current_task}

General project info:
{info}

If needed, full spec is at start.md
DO NOT write code

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "status": "FAIL|PASS",
    "stacktrace": "failure trace, empty if passed",
    "reason": "explanation of failure, empty if passed"
}}

{memory.inject(node_name)}
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
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        passed = False
        evaluation_ended = False
        try:
            status = json.loads(cleaned_response.content)
            passed = status.get("status", "").strip().upper() == "PASS"
            evaluation_ended = isinstance(status, dict)
        except Exception:
            pass

        if evaluation_ended:
            if not passed:
                memory.send_message(cleaned_response.model_copy(), node_name, 'programmer')
            memory.clear_l1(node_name)

        reflection_count = state.get("reflection_count", 0)
        return {
            "active_node": node_name,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "passed": passed,
            "reflection_count": reflection_count + 1 if evaluation_ended else reflection_count,
            "timeout": False
        }

    except asyncio.TimeoutError:
        print("EVALUATOR TIMEOUT")
        return {"passed": False, "node_completed": False, "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"timeout": False, "node_completed": False, "model_error": True}


async def compactor_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = state['active_node']
    tasks_list = state.get('tasks', [])
    current_task = tasks_list[0] if tasks_list else {}

    systemprompt = """
You are a helpful assistant that compacts the conversation history to save tokens, keeping only the most relevant information.
"""
    humanmessage = f"""
Compact this conversation history, keeping only the most relevant information for the current task and removing redundant or irrelevant details.
The conversation belongs to an agent in a software development loop.

the current task:
{current_task}

the log:
{memory.inject(node_name)}
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
        cleaned_response = remove_think_from_message(response)

        memory.compact_l1(cleaned_response, node_name)

        return {
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False,
        }

    except asyncio.TimeoutError:
        print("compactor TIMEOUT")
        return {"passed": False, "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"timeout": False, "node_completed": False, "model_error": True}
    


async def memory_operator_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    tasks_list = state.get('tasks', [])

    finished = len(tasks_list) == 0
    current_task = tasks_list.pop(0) if not finished else {}

    memory.clear_l1('programmer')

    print(f'MEMORY OPERATOR entered, tasks remaining: {len(tasks_list)}')

    return {
        "tasks": tasks_list,
        "finished": finished
    }


async def tool_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    active = state["active_node"]
    last_message = memory.l1.get(active, [])[-1]
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

    for result in results:
        memory.l1.setdefault(active, []).append(result)

    return {}