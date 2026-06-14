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

programmer_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
evaluator_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)
compactor_model = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=minimax_api_key, base_url=BASE_URL)

cg_tools = tools.codegraph_tools

all_tools = [tools.read_file, tools.bash, tools.write_file, tools.str_replace]
tools_by_name = {tool.name: tool for tool in all_tools}

worker_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]
evaluator_tools = [tools.read_file, tools.bash, *cg_tools]
planner_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]

programmer_model = programmer_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)

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



async def programmer_node(state: AgentState):
    systemprompt = """
You are an expert software programmer.
"""
    humanmessage = f"""
According to the start.md in the workspace, implement the entire project as per the requirements specified in the document, ensuring that the final product can be directly run in the current directory. The running requirements should comply with the <API Usage Guide> section of the document. Please complete this task step by step.
Reply with {{"status": "DONE"}} when you are done, JUST THE JSON, no additional text or markdown fences.

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


async def evaluator_node(state: AgentState):
    systemprompt = """
You are an expert QA evaluator.
"""
    humanmessage = f"""
Determine if the programmer has completed the codebase exactly as specified in start.md, making sure
file structure, implementation and tests mirror the spec.
Be strict.

Respond with raw JSON only. No markdown, no code fences, no explanation.

Example:
{{
    "status": "FAIL|PASS",
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
        evaluation_ended = False
        try:
            status = json.loads(cleaned_content)
            passed = status.get("status", "").strip().upper() == "PASS"
            evaluation_ended = isinstance(status, dict)
        except Exception as e:
            pass
        
        response_copy = cleaned_response.model_copy()
        response_copy.content = "The evaluator says: " + cleaned_content

        node_messages = {
            **state["node_messages"],
            "evaluator": state["node_messages"].get("evaluator", []) + [cleaned_response] if not evaluation_ended else [],
        }
        
        reflection_count = state.get("reflection_count", 0)
        return {
            "active_node": "evaluator",
            "node_messages": node_messages,
            "input_tokens": state.get("input_tokens", 0) + usage.get("prompt_tokens", 0),
            "output_tokens": state.get("output_tokens", 0) + usage.get("completion_tokens", 0),
            "passed": passed,
            "reflection_count": reflection_count + 1 if evaluation_ended else reflection_count,
            "timeout": False
        }
    except asyncio.TimeoutError:
        print("EVALUATOR TIMEOUT")
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
