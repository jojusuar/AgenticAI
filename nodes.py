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
    blame: str
    task: str
    timeout: bool
    model_error: bool
    finished: bool
    node_completed: bool
    active_node: Literal["testwriter", "programmer", "planner", "evaluator"]


MODEL_NAME = "MiniMax-M3"
BASE_URL = "https://api.minimax.io/v1"

testwriter_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
programmer_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
evaluator_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
planner_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
entrypoint_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
compactor_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)

cg_tools = tools.codegraph_tools

all_tools = [tools.read_file, tools.bash, tools.write_file, tools.str_replace]
tools_by_name = {tool.name: tool for tool in all_tools}

worker_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]
evaluator_tools = [tools.read_file, tools.bash, *cg_tools]
planner_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]

testwriter_model = testwriter_model.bind_tools(worker_tools)
programmer_model = programmer_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)
planner_model = planner_model.bind_tools(planner_tools)
entrypoint_model = entrypoint_model.bind_tools(planner_tools)

def get_content(message) -> str:
    if isinstance(message.content, str):
        return message.content

    if isinstance(message.content, list):
        return "".join(
            block.get("text", "")
            for block in message.content
            if isinstance(block, dict)
        )
    return ""


def remove_think_from_content(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def remove_think_from_message(message: AIMessage) -> AIMessage:
    cleaned_content = remove_think_from_content(get_content(message))
    msg = message.model_copy()
    msg.content = cleaned_content
    return msg


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


async def entrypoint_node(state: AgentState):
    systemprompt = """
You are a reliable assistant.
You DON'T implement code.
"""
    humanmessage = f"""
Read start.md.

Create specification/.

Split the specification into logically independent markdown files.

Requirements:
- Preserve ALL information.
- Do NOT summarize.
- Do NOT omit details.
- Each requirement must appear exactly once.
- Group related requirements together.
- Create an index.md listing all generated files and their contents.

finally, delete start.md and make sure it does not exist anymore.
Reply {{"status":"DONE"}} when finished.

Log: {format_messages(state, 'entrypoint')}
"""
    print(f'entry PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            entrypoint_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=300
        )
        usage = response.response_metadata.get("token_usage", {})
        cleaned_response = remove_think_from_message(response)
        node_completed = False
        try:
            status = json.loads(remove_think_from_content(get_content(response)))
            node_completed = status['status'] == 'DONE'
        except Exception as e:
            pass
        
        return {
            "active_node": "entrypoint",
            "node_messages": {
                **state["node_messages"],
                "entrypoint": state["node_messages"].get("entrypoint", []) + [cleaned_response] if not node_completed else []
            },
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("entrynode TIMEOUT")
        return {"passed": False, 
                "node_completed": False,
                "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }


async def planner_node(state: AgentState):
    systemprompt = """
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.
"""

    humanmessage = f"""
Based on the specification/ and the current workspace state, do:
1. Write store.json with the immediate current task (schema below)
2. Reply with {{"status": "PROJECT_DONE"}} when the codebase is exactly
as specified in start.md and you've validated JSON syntax is correct, it is mandatory that every file mentioned must exist
in the workspace. 
3. If the workspace is still not finished, update the plan as needed and reply {{"status": "PLANNING_DONE"}}, JUST THE JSON, no additional text.

store.json schema:
{{
    "project_info": {{}},
    "already_done": "description of completed tasks",
    "current_task": {{
        "implementation_spec": "Programmer instructions, DONT REDIRECT TO THE SPEC FILE, write everything needed to know here.",
        "tests_spec": "Testwriter instructions, DONT REDIRECT TO THE SPEC FILE, write everything needed to know here.",
        "test_steps": "What the evaluator must do to run the tests",
        "mentioned_files": []
    }},
    "file_structure": {{}}
}}

Task rules:

1. A task may modify at most:
   - 1 source file
   - 1 test file

2. A task may introduce at most:
   - 1 class, OR
   - 1 enum, OR
   - 3 closely related functions

3. A task must be independently testable.
   The evaluator must be able to determine PASS/FAIL
   without requiring future tasks.

4. NEVER assign an entire module as a task.

Bad:
- Implement all of module/_file.py

Good:
- Add ArguablyException and ArguablyWarning
- Add NoDefault and NO_DEFAULT
- Add InputMethod enum
- Add normalize_action_input()
- Add camel_case_to_kebab_case()
- Add split_unquoted()

5. implementation_spec must contain only the information required
   for the current task, not the entire module specification.

6. mentioned_files must contain only the files needed
   for the current task.

Log: {format_messages(state, 'planner')}
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

        cleaned_response = remove_think_from_message(response)
        cleaned_content = cleaned_response.content
        try:
            cleaned_content = json.loads(cleaned_content)
        except Exception as e:
            pass
        if isinstance(cleaned_content, dict):
            print(get_content(response))
            if cleaned_content.get("status") == "PLANNING_DONE":
                try:
                    with open("workspace/store.json") as f:
                        plan = json.load(f)

                    task = plan.get("current_task", "")
                    node_completed = True

                except Exception as e:
                    print(f"Invalid store.json: {e}")

                    return {
                        "active_node": "planner",
                        "node_messages": {
                            **state["node_messages"],
                            "planner": (
                                state["node_messages"].get("planner", [])
                                + [
                                    HumanMessage(
                                        content=f"store.json contains invalid JSON: {e}. Fix it."
                                    )
                                ]
                            ),
                        },
                        "timeout": False,
                    }
            if cleaned_content.get('status', '') == "PROJECT_DONE":
                finished = True

        return {
            "active_node": "planner",
            "node_messages": {
                **state["node_messages"],
                "planner": state["node_messages"].get("planner", []) + [cleaned_response]
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
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }


async def programmer_node(state: AgentState):
    systemprompt = """
You are an expert software programmer.
"""
    humanmessage = f"""
Your task:
{state.get("task", {}).get('implementation_spec', "no task yet")}

Allowed files:
{state.get("task", {}).get('mentioned_files', '')}

Do the implementation spec and ONLY that.
DO NOT run the tests, an evaluator will do it. Make corrections based on feedback from the evaluator.
You are COMPLETELY FORBIDDEN from reading ANYTHING outside the allowed files list.
Reply with {{"status": "DONE"}} when you are done, JUST THE JSON, no additional text.

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
        cleaned_response = remove_think_from_message(response)
        node_completed = False
        try:
            status = json.loads(remove_think_from_content(get_content(response)))
            node_completed = status['status'] == 'DONE'
        except Exception as e:
            pass
        
        return {
            "active_node": "programmer",
            "node_messages": {
                **state["node_messages"],
                "programmer": state["node_messages"].get("programmer", []) + [cleaned_response] if not node_completed else []
            },
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("programmer TIMEOUT")
        return {"passed": False, 
                "node_completed": False,
                "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }



async def testwriter_node(state: AgentState):
    systemprompt = """
You are an expert software tester.
"""
    humanmessage = f"""
Your task:
{state.get("task", {}).get('tests_spec', "no task yet")}

Allowed files:
{state.get("task", {}).get('mentioned_files', '')}

Do the tests spec and ONLY that.
DO NOT run the tests, an evaluator will do it. Make corrections based on feedback from the evaluator.
You are COMPLETELY FORBIDDEN from reading ANYTHING outside the allowed files list.
Reply with {{"status": "DONE"}} when you are done. JUST THE JSON, no additional text.

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
        cleaned_response = remove_think_from_message(response)

        node_completed = False
        try:
            status = json.loads(remove_think_from_content(get_content(response)))
            node_completed = status['status'] == 'DONE'
        except Exception as e:
            pass
        return {
            "active_node": "testwriter",
            "node_messages": {
                **state["node_messages"],
                "testwriter": state["node_messages"].get("testwriter", []) + [cleaned_response] if not node_completed else []
            },
            "node_completed": node_completed,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("testwriter TIMEOUT")
        return {"passed": False, 
                "node_completed": False,
                "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }


async def evaluator_node(state: AgentState):
    systemprompt = """
You are an expert QA evaluator.
For vitest, always use 'npx vitest run' or 'npm test -- --run' to avoid watch mode blocking.
"""
    humanmessage = f"""
Current task:
{state.get("task", "No tasks yet.")}

Follow the testing instructions.
Determine if the task has been completed, and whose blame it is if not.
You are COMPLETELY FORBIDDEN from reading ANYTHING outside the allowed files list.

Example response (ONLY VALID JSON):
{{
    "status": "FAIL|PASS",
    "blame": "PROGRAMMER|TESTWRITER|(empty if passed)",
    "stacktrace": "failure trace, empty if passed",
    "reason": "explanation of failure, empty if passed"
}}

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
        cleaned_response = remove_think_from_message(response)
        cleaned_content = cleaned_response.content
        
        passed = False
        blame = ""
        try:
            status = json.loads(cleaned_content)
            passed = status.get("status", "").strip().upper() == "PASS"
            blame = status.get("blame", "").strip().lower()
            if blame not in ("programmer", "testwriter"):
                blame = ""
        except Exception as e:
            pass
        evaluation_ended = passed or (blame != "")
        
        response_copy = cleaned_response.model_copy()
        response_copy.content = "The evaluator says: " + cleaned_content

        node_messages = {
            **state["node_messages"],
            "evaluator": state["node_messages"].get("evaluator", []) + [cleaned_response] if not evaluation_ended else [],
        }
        if blame:
            node_messages[blame] = (
                state["node_messages"].get(blame, []) + [response_copy]
            )
        return {
            "active_node": "evaluator",
            "node_messages": node_messages,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "passed": passed,
            "blame": blame,
            "reflection_count": state.get("reflection_count", 0) + 1,
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("EVALUATOR TIMEOUT")
        return {"passed": False, 
                "node_completed": False,
                "blame": '',
                "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }



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
        cleaned_response = remove_think_from_message(response)
        active_node = state['active_node']
        return {
            "node_messages": {
                **state["node_messages"],
                active_node: [cleaned_response]
            },
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "timeout": False,
        }
    except asyncio.TimeoutError:
        print("compactor TIMEOUT")
        return {"passed": False, "timeout": True}
    except Exception as e:
        print(f"MODEL ERROR: {e}")

        return {
            "timeout": False,
            "node_completed": False,
            "model_error": True
        }



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
        "blame": "",
        "reflection_count": 0,
        "node_completed": False
    }