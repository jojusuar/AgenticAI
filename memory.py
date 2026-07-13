import re
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage
)


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


class MyMemory:
    l1: dict
    l2: dict
    l3: dict

    def __init__(self):
        self.l1 = {}
        self.l2 = {}
        self.l3 = {}

    def add_self_message(self, message: AIMessage, node: str):
        content = get_content(message)
        content = f'{node.upper()} (YOU): {content}'
        message.content = content
        self.l1.setdefault(node, []).append(message)

    def send_message(self, message: AnyMessage, source_node: str, target_node: str):
        content = get_content(message)
        content = f'{source_node.upper()}: {content}'
        message.content = content
        self.l1.setdefault(target_node, []).append(message)

    def compact_l1(self, compacted: AIMessage, node: str):
        self.l1[node] = [compacted]

    def format_messages(self, node: str) -> str:
        messages = self.l1.get(node, [])
        if not messages:
            return ""
        lines = []
        for msg in messages:
            content = get_content(msg)
            if isinstance(msg, HumanMessage):
                lines.append(f"USER: {content}")
            elif isinstance(msg, AIMessage):
                if content:
                    lines.append(content)
                for tc in msg.tool_calls:
                    args = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                    lines.append(f"{node.upper()} (YOU) called tool `{tc['name']}({args})`")
            elif isinstance(msg, ToolMessage):
                lines.append(f"TOOL RESULT:\n{content}")
        return "\n\n".join(lines)

    def inject(self, node: str) -> str:
        return f'''Your log (last is most recent):\n{self.format_messages(node)}'''

    def clear_l1(self, node: str):
        self.l1[node] = []