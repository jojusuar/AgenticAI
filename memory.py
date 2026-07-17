import math
import os
import re
from typing import Any

try:
    import ollama
except Exception:
    ollama = None
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage
)


class OllamaEmbeddingError(RuntimeError):
    pass


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


class MonolithicLog(dict):
    def __init__(self, shared_key: str):
        super().__init__()
        self.shared_key = shared_key
        super().__setitem__(shared_key, [])

    def _shared(self):
        return super().__getitem__(self.shared_key)

    def get(self, key, default=None):
        return self._shared()

    def setdefault(self, key, default=None):
        return self._shared()

    def __getitem__(self, key):
        return self._shared()

    def __setitem__(self, key, value):
        super().__setitem__(self.shared_key, value)


class MyMemory:
    l1: dict
    l2: dict
    l3: dict

    def __init__(self, memory: bool = True, l3_insights: int = 5):
        self.memory = memory
        self.monolithic = not memory
        self.l3_insights = l3_insights
        self.l1 = MonolithicLog("all") if self.monolithic else {}
        self.l2 = {}
        self.l3 = []

    def add_self_message(self, message: AIMessage, node: str):
        content = get_content(message)
        content = f'{node.upper()}: {content}'
        message.content = content
        message.additional_kwargs["source_node"] = node
        self.l1.setdefault(node, []).append(message)

    def send_message(self, message: AnyMessage, source_node: str, target_node: str):
        content = get_content(message)
        content = f'{source_node.upper()}: {content}'
        message.content = content
        message.additional_kwargs["source_node"] = source_node
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
            source_node = msg.additional_kwargs.get("source_node", node)
            if isinstance(msg, HumanMessage):
                lines.append(f"USER: {content}")
            elif isinstance(msg, AIMessage):
                if content:
                    lines.append(content)
                for tc in msg.tool_calls:
                    args = ", ".join(f"{k}={repr(v)}" for k, v in tc["args"].items())
                    lines.append(f"{source_node.upper()} called tool `{tc['name']}({args})`")
            elif isinstance(msg, ToolMessage):
                prefix = f"TOOL RESULT for {source_node.upper()}" if self.monolithic else "TOOL RESULT"
                lines.append(f"{prefix}:\n{content}")
        return "\n\n".join(lines)

    def inject(self, node: str, task: dict[str, Any] | None = None) -> str:
        label = "Shared log" if self.monolithic else "Your log"
        l1 = f'''{label} (last is most recent):\n{self.format_messages(node)}'''
        if self.monolithic or task is None:
            return l1

        return "\n\n".join(
            section
            for section in (
                self.inject_l2(task, node),
                self.inject_l3(task, node),
                l1,
            )
            if section.strip()
        )

    def context_for_task(self, task: dict[str, Any], node: str) -> str:
        return self.inject(node, task)

    def clear_l1(self, node: str):
        if self.monolithic:
            return
        self.l1[node] = []

    def clear_all_l1(self):
        if self.monolithic:
            return
        self.l1 = {}

    def relevant_files_for_task(self, task: dict[str, Any]) -> list[str]:
        files = []
        for key in ("target_module", "relevant_files"):
            value = task.get(key)
            if isinstance(value, str):
                files.append(value)
            elif isinstance(value, list):
                files.extend(item for item in value if isinstance(item, str))
        return list(dict.fromkeys(files))

    def update_l2(self, path: str, summary: str):
        if summary.strip():
            self.l2[path] = summary.strip()

    def inject_l2(self, task: dict[str, Any], node: str) -> str:
        if node == "planner":
            if not self.l2:
                return ""
            summaries = [
                f"{path}:\n{summary}"
                for path, summary in self.l2.items()
            ]
            return (
                "Known module memory from completed tasks. Prefer these summaries over reading full files unless exact code is necessary:\n"
                + "\n\n".join(summaries)
            )

        files = self.relevant_files_for_task(task)
        summaries = [
            f"{path}:\n{self.l2[path]}"
            for path in files
            if path in self.l2
        ]
        if not summaries:
            return ""
        return (
            "Module memory for files relevant to the current task. Prefer these summaries over reading full files unless exact code is necessary:\n"
            + "\n\n".join(summaries)
        )

    def embed_text(self, text: str) -> list[float]:
        if ollama is None:
            raise OllamaEmbeddingError("Ollama Python package is not installed")

        host = os.environ.get("OLLAMA_HOST")
        try:
            client = ollama.Client(host=host) if host else ollama.Client()
            response = client.embeddings(model="bge-m3", prompt=text)
        except Exception as e:
            host_label = host or "default Ollama host"
            raise OllamaEmbeddingError(f"Ollama embedding request failed using {host_label}: {e}") from e
        if isinstance(response, dict):
            embedding = response.get("embedding")
        else:
            embedding = getattr(response, "embedding", None)

        if not isinstance(embedding, list) or not embedding:
            raise OllamaEmbeddingError(f"Unexpected Ollama embedding response: {response!r}")

        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as e:
            raise OllamaEmbeddingError(f"Ollama embedding contains non-numeric values: {embedding!r}") from e

    def insert_l3(self, abstract: str, insight: str):
        abstract = abstract.strip()
        insight = insight.strip()
        if not abstract or not insight:
            return
        embedding = self.embed_text(abstract)
        if not embedding:
            return
        self.l3.append({
            "abstract": abstract,
            "insight": insight,
            "embedding": embedding,
        })

    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def retrieve_l3(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or not self.l3:
            return []
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        ranked = sorted(
            self.l3,
            key=lambda item: self.cosine_similarity(query_embedding, item["embedding"]),
            reverse=True,
        )
        return ranked[:k]

    def inject_l3(self, task: dict[str, Any], node: str, k: int | None = None) -> str:
        if k is None:
            k = self.l3_insights
        if not self.l3:
            print("MEMORY INJECT L3 SKIPPED (stored=0)")
            return ""
        if node == "planner":
            print(f"MEMORY INJECT L3 ALL {len(self.l3)} for planner")
            lines = [
                f"- {item['insight']}"
                for item in self.l3
            ]
            return "Known project memory insights from completed tasks:\n" + "\n".join(lines)
        query = "\n".join(
            str(task.get(key, ""))
            for key in ("task", "test_instructions", "target_module")
        )
        insights = self.retrieve_l3(query, k=k)
        if not insights:
            print(f"MEMORY INJECT L3 SKIPPED (stored={len(self.l3)}, retrieved=0)")
            return ""
        print(f"MEMORY INJECT L3 RETRIEVED {len(insights)} of {len(self.l3)}")
        lines = [
            f"- {item['insight']}"
            for item in insights
        ]
        return "Project memory insights relevant to the current task:\n" + "\n".join(lines)
