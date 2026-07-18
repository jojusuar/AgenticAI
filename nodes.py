import asyncio
import hashlib
import json
import re
import time
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
from memory import MyMemory, OllamaEmbeddingError, remove_think_from_message


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
    last_workspace_change_time: float
    idle_timeout: bool
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

    for key in ("test_instructions", "target_files", "relevant_files", "target_file_structure"):
        if key not in current_task and key in cleaned:
            current_task[key] = cleaned[key]

    return current_task, project_info, finished


def get_token_usage(response):
    metadata = response.response_metadata or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage_metadata.get("input_tokens")
        or usage_metadata.get("prompt_tokens")
        or 0
    )
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage_metadata.get("output_tokens")
        or usage_metadata.get("completion_tokens")
        or 0
    )
    return input_tokens, output_tokens


async def planner_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    node_name = 'planner'
    current_task = state.get('current_task', {})

    prompt = f"""
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.

Plan only the immediate next implementation task needed to satisfy the specification at start.md.
Do not create a task list. Do not plan future tasks beyond the next one.
Each task must represent one coherent, independently evaluable change.
Prefer a single target file, but allow up to five target files when all of them are necessary for the same feature, integration boundary, or scaffolding operation.
Scaffolding tasks may group closely related package metadata, directory initializers, entry points, and minimal test structure.
Do not group unrelated functionality into one task. Every target file must be justified by the task description.
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
        "target_files": ["One to five files the programmer must create or modify for this task."],
        "relevant_files": ["Existing files useful for context but not intended to be modified by this task."],
        "target_file_structure": {{
                                    "folder1": {{
                                        "subfolder1": ["file1.py", "file2.py"],
                                        ...
                                    }},
                                    ...
                                }}
    }}
}}

{memory.inject(node_name, current_task)}
"""
    print(f'PLANNER PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(planner_model, prompt),
            timeout=300
        )
        input_tokens, output_tokens = get_token_usage(response)
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)
        
        node_completed = False

        cleaned = parse_json_response(cleaned_response.content)
        if cleaned is PARSE_FAILED:
            return {
                "active_node": node_name,
                "input_tokens": state.get("input_tokens", 0) + input_tokens,
                "output_tokens": state.get("output_tokens", 0) + output_tokens,
                "node_completed": False,
                "timeout": False,
                "model_error": False
            }

        current_task, project_info, finished = normalize_planner_response(cleaned)
        valid_task = (
            isinstance(current_task, dict)
            and current_task.get('task')
            and isinstance(current_task.get('target_files'), list)
            and 1 <= len(current_task.get('target_files')) <= 5
            and all(
                isinstance(path, str) and bool(path.strip())
                for path in current_task.get('target_files')
            )
            and len(set(current_task.get('target_files'))) == len(current_task.get('target_files'))
            and isinstance(current_task.get('relevant_files'), list)
            and all(
                isinstance(path, str) and bool(path.strip())
                for path in current_task.get('relevant_files')
            )
            and len(set(current_task.get('relevant_files'))) == len(current_task.get('relevant_files'))
            and not set(current_task.get('target_files')).intersection(current_task.get('relevant_files'))
        )
        if (finished or valid_task) and project_info:
            node_completed = True
        else:
            memory.add_self_message(HumanMessage(content='The response JSON does not follow the schema.'), node_name)

        planner_retries = 0 if node_completed else state.get("planner_retries", 0) + 1

        return {
            "active_node":node_name,
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
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
    prompt = f"""
You are an expert software programmer.

The current task:
{current_task}

General project info:
{info}

If needed, full spec is at start.md
You must fix what the evaluator tells you to if it gives feedback.
    
Reply with {{"status": "DONE"}} when you are done, JUST THE JSON, no additional text or markdown fences.

{memory.inject(node_name, current_task)}
"""
    print(f'{node_name} PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(programmer_model, prompt),
            timeout=300
        )

        input_tokens, output_tokens = get_token_usage(response)
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        node_completed = False
        status = parse_json_response(cleaned_response.content)
        if status is PARSE_FAILED:
            return {
                "active_node": node_name,
                "node_completed": False,
                "input_tokens": state.get("input_tokens", 0) + input_tokens,
                "output_tokens": state.get("output_tokens", 0) + output_tokens,
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
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
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

{memory.inject(node_name, current_task)}
"""
    print(f'EVALUATOR PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(evaluator_model, prompt),
            timeout=300
        )
        input_tokens, output_tokens = get_token_usage(response)
        cleaned_response = remove_think_from_message(response)
        memory.add_self_message(cleaned_response.model_copy(), node_name)

        passed = False
        evaluation_ended = False
        status = parse_json_response(cleaned_response.content)
        if status is PARSE_FAILED:
            return {
                "active_node": node_name,
                "input_tokens": state.get("input_tokens", 0) + input_tokens,
                "output_tokens": state.get("output_tokens", 0) + output_tokens,
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
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
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
{memory.inject(node_name, current_task)}
"""
    print(f'COMPACTOR PROMPT: {prompt}')
    try:
        response = await asyncio.wait_for(
            invoke_model(compactor_model, prompt),
            timeout=300
        )
        input_tokens, output_tokens = get_token_usage(response)
        cleaned_response = remove_think_from_message(response)

        memory.compact_l1(cleaned_response, node_name)

        return {
            "input_tokens": state.get("input_tokens", 0) + input_tokens,
            "output_tokens": state.get("output_tokens", 0) + output_tokens,
            "timeout": False,
            "model_error": False,
        }

    except asyncio.TimeoutError:
        print("compactor TIMEOUT")
        return {"active_node": node_name, "passed": False, "timeout": True, "model_error": False}
    except Exception as e:
        print(f"MODEL ERROR: {e}")
        return {"active_node": node_name, "timeout": False, "node_completed": False, "model_error": True}
    


async def l2_operator_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    current_task = state.get('current_task', {})
    input_tokens_total = state.get("input_tokens", 0)
    output_tokens_total = state.get("output_tokens", 0)

    if memory.monolithic:
        return {}

    touched_files = sorted(memory.programmer_touched_files)
    for path in touched_files:
        try:
            data = tools.safe_path(path).read_bytes()
            content_hash = hashlib.sha256(data).hexdigest()
            content = data.decode("utf-8")
        except (OSError, UnicodeError) as e:
            print(f"MEMORY OPERATOR L2 READ ERROR for {path}: {e}")
            continue

        if memory.l2_hashes.get(path) == content_hash:
            memory.mark_l2_unchanged(path)
            print(f"MEMORY OPERATOR L2 UNCHANGED {path}")
            continue

        prompt = f"""
You are the memory operator for an agentic code-generation harness.
Summarize the current implementation of this file for future tasks.

Focus on exposed interfaces, imports/dependencies, classes, functions, method signatures, exports, side effects, and factual constraints imposed by this file's current code or API.
Describe only the file as it exists. Do not give advice, recommendations, implementation plans, or instructions to future agents.
Do not include irrelevant prose. Keep it concise but complete.

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "summary": "Concise file summary for future tasks."
}}

Current task:
{current_task}

File path:
{path}

File content:
{content}
"""
        try:
            response = await asyncio.wait_for(
                invoke_model(memory_operator_model, prompt),
                timeout=300
            )
            input_tokens, output_tokens = get_token_usage(response)
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            cleaned_response = remove_think_from_message(response)
            parsed = parse_json_response(cleaned_response.content)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str):
                print(f"MEMORY OPERATOR L2 MALFORMED JSON for {path}")
                continue
            summary = parsed["summary"]
            memory.complete_l2_update(path, content_hash, summary)
            print(f"MEMORY OPERATOR L2 UPDATED {path}")
        except asyncio.TimeoutError:
            print(f"MEMORY OPERATOR L2 TIMEOUT for {path}")
        except Exception as e:
            print(f"MEMORY OPERATOR L2 ERROR for {path}: {e}")

    return {
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
    }


async def l3_operator_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    current_task = state.get('current_task', {})
    input_tokens_total = state.get("input_tokens", 0)
    output_tokens_total = state.get("output_tokens", 0)

    if memory.monolithic or not state.get("passed"):
        return {}

    task_log = memory.format_messages('programmer')
    insight_prompt = f"""
You are the L3 memory operator for an agentic code-generation harness.

Extract at most one durable project-level insight from the completed task.

L3 memory is for cross-task knowledge only. Store an insight only if it is likely to help future tasks across multiple files or modules.

Good L3 insights include:
- Cross-module architecture or API conventions.
- Non-obvious integration constraints between files.
- Stable project-specific testing or validation lessons.
- Repeated failure modes or gotchas observed during implementation/evaluation.
- Decisions that future tasks must preserve for consistency.

Do NOT store:
- A summary of the completed task.
- A summary of one file's functions/classes/imports.
- Implementation details already suitable for L2 module memory.
- General software advice.
- Benchmark-specific strategy or knowledge about hidden evaluation.
- Anything that is only useful for the exact task that just finished.

If there is no cross-task insight, return an empty string.

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "insight": "One concise cross-task project insight, or empty string if none."
}}

Completed task:
{current_task}

Programmer L1 (includes evaluator feedback sent to the programmer):
{task_log}
"""
    try:
        print(
            "MEMORY OPERATOR L3 START "
            f"(programmer_l1_chars={len(task_log)}, stored={len(memory.l3)})"
        )
        insight_response = await asyncio.wait_for(
            invoke_model(memory_operator_model, insight_prompt),
            timeout=300
        )
        input_tokens, output_tokens = get_token_usage(insight_response)
        input_tokens_total += input_tokens
        output_tokens_total += output_tokens
        cleaned_insight = remove_think_from_message(insight_response)
        parsed_insight = parse_json_response(cleaned_insight.content)
        if not isinstance(parsed_insight, dict) or not isinstance(parsed_insight.get("insight"), str):
            print("MEMORY OPERATOR L3 INSIGHT MALFORMED JSON")
            parsed_insight = {"insight": ""}
        insight = parsed_insight["insight"].strip()
        if not insight:
            print("MEMORY OPERATOR L3 NO CROSS-TASK INSIGHT")
        else:
            abstract_prompt = f"""
Summarize this L3 project insight as one concise retrieval query sentence.

Preserve only the concepts needed to retrieve the insight for future related tasks.
Do not add new facts.

Expected response schema (raw JSON only. No markdown, no code fences, no explanation):
{{
    "abstract": "Concise semantic retrieval sentence."
}}

Insight:
{insight}
"""
            abstract_response = await asyncio.wait_for(
                invoke_model(memory_operator_model, abstract_prompt),
                timeout=300
            )
            input_tokens, output_tokens = get_token_usage(abstract_response)
            input_tokens_total += input_tokens
            output_tokens_total += output_tokens
            cleaned_abstract = remove_think_from_message(abstract_response)
            parsed_abstract = parse_json_response(cleaned_abstract.content)
            if not isinstance(parsed_abstract, dict) or not isinstance(parsed_abstract.get("abstract"), str):
                print("MEMORY OPERATOR L3 ABSTRACT MALFORMED JSON")
                abstract = insight
            else:
                abstract = parsed_abstract["abstract"].strip() or insight
            memory.insert_l3(abstract, insight)
            print(f"MEMORY OPERATOR L3 STORED ({len(memory.l3)} total)")
    except asyncio.TimeoutError:
        print("MEMORY OPERATOR L3 TIMEOUT")
    except OllamaEmbeddingError:
        raise
    except Exception as e:
        print(f"MEMORY OPERATOR L3 ERROR: {e}")

    memory.clear_all_l1()

    print('MEMORY OPERATOR L3 entered, rerouting to planner')

    return {
        "passed": False,
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total
    }


async def task_cleanup_node(state: AgentState):
    """Discard task-scoped L1 after an abandoned task without creating L3 memory."""
    memory: MyMemory = state.get('memory')
    memory.clear_all_l1()
    return {"passed": False}


async def tool_node(state: AgentState):
    memory: MyMemory = state.get('memory')
    active = state["active_node"]
    last_message = memory.l1.get(active, [])[-1]
    results = []
    workspace_changed = False

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        try:
            tool = tools_by_name[tool_name]
            observation = await tool.ainvoke(tool_call["args"])
            if tool_name in {"write_file", "str_replace"} and not str(observation).startswith("Error:"):
                workspace_changed = True
                if active == "programmer":
                    path = tool_call.get("args", {}).get("path")
                    if isinstance(path, str):
                        memory.track_programmer_file(path)
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

    if workspace_changed:
        return {"last_workspace_change_time": time.monotonic(), "idle_timeout": False}
    return {}
