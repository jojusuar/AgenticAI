import asyncio
from typing import Literal
from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from typing_extensions import TypedDict, Annotated
import operator
import tools
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

def merge_node_messages(a: dict, b: dict) -> dict:
    result = dict(a)
    for key, msgs in b.items():
        result[key] = result.get(key, []) + msgs
    return result


class AgentState(TypedDict):
    node_messages: Annotated[dict, merge_node_messages]

    reflection_count: Annotated[int, operator.add]
    passed: bool

    start_time: float
    elapsed_time: float

    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]

    fixing: bool
    task_id: str
    active_node: Literal["worker1", "worker2", "planner", "test_evaluator"]


MODEL_NAME = "gemma-4-31b-it"

worker1_model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key
)
worker2_model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key
)
evaluator_model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key
)
planner_model = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    api_key=api_key
)

all_tools = [tools.read_file, tools.bash, tools.write_file, tools.str_replace]
tools_by_name = {tool.name: tool for tool in all_tools}

worker_tools = [tools.read_file, tools.write_file, tools.str_replace, tools.bash]
evaluator_tools = [tools.read_file, tools.bash]
planner_tools = [tools.read_file, tools.write_file, tools.str_replace]

worker1_model = worker1_model.bind_tools(worker_tools)
worker2_model = worker2_model.bind_tools(worker_tools)
evaluator_model = evaluator_model.bind_tools(evaluator_tools)
planner_model = planner_model.bind_tools(planner_tools)

def get_content(response) -> str:
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list) and response.content:
        return response.content[0].get("text", "")
    return ""

def format_messages(state: AgentState, node: str, char_limit: int = 3000) -> str:
    messages = state.get("node_messages", {}).get(node, [])
    lines = []
    for msg in messages:
        content = get_content(msg)
        content = content[:char_limit] + "[TRUNCATED]" if len(content) > char_limit else content
        if isinstance(msg, HumanMessage):
            lines.append(f"USER: {content}")
        elif isinstance(msg, AIMessage):
            if content:
                lines.append(f"ASSISTANT: {content}")
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                lines.append(f"ASSISTANT called tool `{tc['name']}({args})`")
        elif isinstance(msg, ToolMessage):
            lines.append(f"TOOL RESULT:\n{content}")
    return "\n\n".join(lines)


async def planner_node(state: AgentState):
    systemprompt = """
You are a senior software engineer who plans and manages tasks in the codebase.
You DON'T implement code.
Project planning and descriptions must be stored in store.json
"""
    humanmessage = f"""
Your log:
{format_messages(state, 'planner')}
"""
    print(f'PLANNER PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            planner_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=60
        )
        usage = response.response_metadata
        return {
            "active_node": "planner",
            "node_messages": {"planner": [response]},
            "input_tokens": usage.get("prompt_eval_count", 0),
            "output_tokens": usage.get("eval_count", 0),
        }
    except asyncio.TimeoutError:
        return {"passed": False}


async def worker2_node(state: AgentState):
    previous_agent_log = state['node_messages'].get(state.get('active_node'), [])
    systemprompt = """
You are an expert software programmer.
You have access to tools for reading and writing files.
"""
    humanmessage = f"""
Read the project info at store.json and select the first pending task, execute it.

Message from previous agent:
{previous_agent_log[-1].content if previous_agent_log else ''}
    
Your log:
{format_messages(state, 'worker2')}
"""
    print(f'WORKER2 PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            worker2_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=180
        )
        usage = response.response_metadata
        return {
            "active_node": "worker2",
            "node_messages": {"worker2": [response]},
            "input_tokens": usage.get("prompt_eval_count", 0),
            "output_tokens": usage.get("eval_count", 0),
        }
    except asyncio.TimeoutError:
        return {"passed": False}


async def worker1_node(state: AgentState):
    active = state.get('active_node')
    previous_agent_log = state['node_messages'].get(active, [])
    systemprompt = """
You are an expert software tester.
"""
    humanmessage = f"""
Read the project info at store.json and select the first pending task, execute it.
ONLY prepare the test code/environment/dependencies for the test, but DO NOT run it.

Message from {active} agent:
{previous_agent_log[-1].content if previous_agent_log else ''}
    
Your log:
{format_messages(state, 'worker1')}
"""
    print(f'WORKER1 PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            worker1_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=180
        )
        usage = response.response_metadata
        return {
            "active_node": "worker1",
            "node_messages": {"worker1": [response]},
            "input_tokens": usage.get("prompt_eval_count", 0),
            "output_tokens": usage.get("eval_count", 0),
        }
    except asyncio.TimeoutError:
        return {"passed": False}


async def test_evaluator_node(state: AgentState):
    previous_agent_log = state['node_messages'].get(state.get('active_node'), [])
    systemprompt = """
You are an expert QA evaluator.
For vitest, always use 'npx vitest run' or 'npm test -- --run' to avoid watch mode blocking.
"""
    humanmessage = f"""
Read the project details at store.json and notice the first pending task.
Follow the test instructions of the task to evaluate its implementation.
If it is done, mark it as done in the store and reply just OK, if it's not then reply
with what failed and a brief explanation.
DO NOT write or modify the code.

Message from previous agent:
{previous_agent_log[-1].content if previous_agent_log else ''}
    
Your log:
{format_messages(state, 'test_evaluator')}
"""
    print(f'EVALUATOR PROMPT: {humanmessage}')
    try:
        response = await asyncio.wait_for(
            evaluator_model.ainvoke([
                SystemMessage(content=systemprompt),
                HumanMessage(content=humanmessage)
            ]),
            timeout=60
        )
        usage = response.response_metadata
        passed = 'OK' in response.content
        return {
            "active_node": "test_evaluator",
            "node_messages": {"test_evaluator": [response]},
            "input_tokens": usage.get("prompt_eval_count", 0),
            "output_tokens": usage.get("eval_count", 0),
            "passed": passed,
            "reflection_count": 1,
        }
    except asyncio.TimeoutError:
        return {"passed": False}


def tool_node(state: AgentState):
    active = state["active_node"]
    last_message = state["node_messages"][active][-1]
    result = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"node_messages": {active: result}}

