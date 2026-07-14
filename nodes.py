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
    current_task: dict
    reflection_count: int
    planner_retries: int
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
planner_tools = [tools.read_file, tools.bash]

programmer_model = programmer_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)
planner_model = planner_model.bind_tools(planner_tools)

PARSE_FAILED = object()


async def invoke_model(model, prompt: str):
    try:
        return await model.ainvoke([
            SystemMessage(content=prompt)
        ])
    except Exception as e:
        if "chat content is empty" not in str(e):
            raise
        return await model.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Proceed.")
        ])


def json_loads_lenient(text: str):
    return json.loads(text, strict=False)


def close_unbalanced_json_object(text: str):
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

    if depth > 0 and not in_string:
        return text + ("}" * depth)
    return text


def parse_json_response(content: str):
    try:
        return json_loads_lenient(content)
    except Exception:
        pass

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    for prefix in (
        "PLANNER (YOU):",
        "PROGRAMMER (YOU):",
        "EVALUATOR (YOU):",
        "PLANNER:",
        "PROGRAMMER:",
        "EVALUATOR:",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned.split(":", 1)[1].strip()
            break

    try:
        return json_loads_lenient(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json_loads_lenient(candidate)
        except Exception:
            pass

        repaired = close_unbalanced_json_object(cleaned[start:])
        if repaired != cleaned[start:]:
            try:
                return json_loads_lenient(repaired)
            except Exception:
                return PARSE_FAILED
        return PARSE_FAILED
    return PARSE_FAILED


def normalize_planner_response(cleaned):
    if not isinstance(cleaned, dict):
        return {}, "", False

    project_info = cleaned.get('project_info', '')
    finished = bool(cleaned.get('finished', False))
    current_task = cleaned.get('task') or {}

    if isinstance(current_task, str):
        current_task = {"task": current_task}
    if not isinstance(current_task, dict):
        current_task = {}

    for key in ("test_instructions", "target_module", "relevant_files", "target_file_structure"):
        if key not in current_task and key in cleaned:
            current_task[key] = cleaned[key]

    if not current_task.get("target_module"):
        relevant_files = current_task.get("relevant_files") or []
        if relevant_files:
            current_task["target_module"] = relevant_files[0]

    return current_task, project_info, finished


async def planner_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'planner'

    prompt = f"""
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.

Plan only the immediate next implementation task needed to satisfy the specification at start.md.
Do not create a task list. Do not plan future tasks beyond the next one.
Each task must be small and must require implementing or modifying only one module.
The task must be self-contained and finishable from the repository state that exists now.
Do not assign a task that depends on code, modules, package metadata, tests, or generated artifacts that have not been created yet.
If needed, choose an earlier enabling task instead.
If the project is already complete, return finished=true and task=null.

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "project_info": "Project abstract, details about language, dependencies needed, frameworks, etc",
    "finished": false,
    "task": {{
        "task": "Short task description, class definitions or function signatures if applicable, and any relevant details for implementation.",
        "test_instructions": "Instructions for the evaluator to test this single task, like running tests, verifying imports, or empirical evaluation, but NO WRITING CODE.",
        "target_module": "The single module/file the programmer should implement or modify for this task.",
        "relevant_files": ["names", "of", "files", "relevant", "to", "this", "task"],
        "target_file_structure": {{
                                    "folder1": {{
                                        "subfolder1": ["file1.py", "file2.py"],
                                        ...
                                    }},
                                    ...
                                }}
    }}
}}

{memory.inject(node_name)}
"""
    print(f'PLANNER PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(planner_model, prompt),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)
        
        node_completed = False

        cleaned = parse_json_response(cleaned_response.content)
        if cleaned is PARSE_FAILED:
            return {
                "active_node": node_name,
                "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
                "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
                "node_completed": False,
                "timeout": False,
                "model_error": False
            }

        current_task, project_info, finished = normalize_planner_response(cleaned)
        valid_task = (
            isinstance(current_task, dict)
            and current_task.get('task')
            and current_task.get('target_module')
        )
        if (finished or valid_task) and project_info:
            node_completed = True
        else:
            memory.add_self_message(HumanMessage(content='The response JSON does not follow the schema.'), node_name)

        planner_retries = 0 if node_completed else state.get("planner_retries", 0) + 1

        return {
            "active_node":node_name,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "node_completed": node_completed,
            "timeout": False,
            "current_task": current_task,
            "project_info": project_info,
            "finished": finished,
            "passed": False,
            "reflection_count": 0,
            "planner_retries": planner_retries,
            "model_error": False
        }
    
    except asyncio.TimeoutError:
        print("PLANNER TIMEOUT")
        return {"active_node": node_name, "passed": False, "node_completed": False, "timeout": True, "model_error": False}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"active_node": node_name, "timeout": False, "node_completed": False, "model_error": True}
    

async def programmer_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'programmer'
    info = state.get('project_info', '')
    current_task = state.get('current_task', {})
    # instructions = current_task.get('task', 'No task available.')
    # evaluator_instructions = current_task.get('test', 'No task available.')
    # relevant_files = current_task.get('relevant_files', [])
    # target_file_structure = current_task.get('target_file_structure', '')

    prompt = f"""
You are an expert software programmer.

The current task:
{current_task}

General project info:
{info}

If needed, full spec is at start.md
You must fix what the evaluator tells you to if it gives feedback.
    
Reply with {{"status": "DONE"}} when you are done, JUST THE JSON, no additional text or markdown fences.

{memory.inject(node_name)}
"""
    print(f'{node_name} PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(programmer_model, prompt),
            timeout=300
        )

        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        node_completed = False
        status = parse_json_response(cleaned_response.content)
        if status is PARSE_FAILED:
            return {
                "active_node": node_name,
                "node_completed": False,
                "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
                "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
                "timeout": False,
                "model_error": False
            }
        if isinstance(status, dict) and status.get('status') == 'DONE':
            node_completed = True
        else:
            memory.add_self_message(HumanMessage(content='The response JSON does not follow the schema.'), node_name)
        
        return {
            "active_node": node_name,
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False,
            "model_error": False
        }

    except asyncio.TimeoutError:
        print(f"{node_name} TIMEOUT")
        return {"active_node": node_name, "passed": False, "node_completed": False, "timeout": True, "model_error": False}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"active_node": node_name, "timeout": False, "node_completed": False, "model_error": True}


async def evaluator_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'evaluator'
    info = state.get('project_info', '')
    current_task = state.get('current_task', {})
    # instructions = current_task.get('test_instructions', 'No task available.')
    # relevant_files = current_task.get('relevant_files', [])
    # target_file_structure = current_task.get('target_file_structure', '')

    prompt = f"""
You are an expert QA evaluator.

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
    print(f'EVALUATOR PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(evaluator_model, prompt),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        passed = False
        evaluation_ended = False
        status = parse_json_response(cleaned_response.content)
        if status is PARSE_FAILED:
            return {
                "active_node": node_name,
                "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
                "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
                "passed": False,
                "node_completed": False,
                "timeout": False,
                "model_error": False
            }

        if isinstance(status, dict) and status.get("status", "").strip().upper() in {"PASS", "FAIL"}:
            passed = status.get("status", "").strip().upper() == "PASS"
            evaluation_ended = True
        else:
            memory.add_self_message(HumanMessage(content='The response JSON does not follow the schema.'), node_name)

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
            "node_completed": evaluation_ended,
            "reflection_count": reflection_count + 1 if evaluation_ended and not passed else reflection_count,
            "timeout": False,
            "model_error": False
        }

    except asyncio.TimeoutError:
        print("EVALUATOR TIMEOUT")
        return {"active_node": node_name, "passed": False, "node_completed": False, "timeout": True, "model_error": False}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"active_node": node_name, "timeout": False, "node_completed": False, "model_error": True}


async def compactor_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = state.get('active_node', 'planner')
    current_task = state.get('current_task', {})

    prompt = f"""
You are a helpful assistant that compacts the conversation history to save tokens, keeping only the most relevant information.

Compact this conversation history, keeping only the most relevant information for the current task and removing redundant or irrelevant details.
The conversation belongs to an agent in a software development loop.

the current task:
{current_task}

the log:
{memory.inject(node_name)}
"""
    print(f'COMPACTOR PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(compactor_model, prompt),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)

        memory.compact_l1(cleaned_response, node_name)

        return {
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False,
            "model_error": False,
        }

    except asyncio.TimeoutError:
        print("compactor TIMEOUT")
        return {"active_node": node_name, "passed": False, "timeout": True, "model_error": False}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"active_node": node_name, "timeout": False, "node_completed": False, "model_error": True}
    


async def memory_operator_node(state: AgentState):
    memory: MyMemory = state.get('memory')

    memory.clear_l1('programmer')

    print('MEMORY OPERATOR entered, rerouting to planner')

    return {
        "passed": False
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
        result = ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        result.additional_kwargs["source_node"] = active
        results.append(result)

    for result in results:
        memory.l1.setdefault(active, []).append(result)

    return {}
