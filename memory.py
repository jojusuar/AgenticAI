import hashlib
import math
import os
import re
from pathlib import PurePosixPath
from typing import Any

import tools

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


def normalize_workspace_path(path: str) -> str | None:
    """Normalize a model/tool path to a safe workspace-relative POSIX key."""
    if not isinstance(path, str) or not path.strip():
        return None
    replaced = path.strip().replace("\\", "/")
    normalized = PurePosixPath(replaced)
    if (
        "\x00" in replaced
        or normalized.is_absolute()
        or re.match(r"^[A-Za-z]:", replaced)
        or ".." in normalized.parts
        or str(normalized) in {"", "."}
    ):
        return None
    return str(normalized)


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

    def __init__(
        self,
        memory: bool | None = None,
        l1_enabled: bool = True,
        l2_enabled: bool = True,
        l3_enabled: bool = True,
        l3_similarity_threshold: float = 0.60,
    ):
        if memory is not None:
            l1_enabled = memory
            l2_enabled = memory
            l3_enabled = memory
        self.l1_enabled = l1_enabled
        self.l2_enabled = l2_enabled
        self.l3_enabled = l3_enabled
        self.memory = l1_enabled or l2_enabled or l3_enabled
        self.monolithic = not l1_enabled
        if not 0.0 <= l3_similarity_threshold <= 1.0:
            raise ValueError("l3_similarity_threshold must be between 0.0 and 1.0")
        self.l3_similarity_threshold = l3_similarity_threshold
        self.l1 = MonolithicLog("all") if self.monolithic else {}
        self.l2 = {}
        self.l2_hashes = {}
        self.programmer_touched_files = set()
        self.l3 = []
        self.l3_next_id = 1
        self.l1_task_checkpoint = 0

    def add_self_message(self, message: AIMessage, node: str):
        content = get_content(message)
        content = f'{node.upper()}: {content}'
        message.content = content
        message.additional_kwargs["source_node"] = node
        self.l1.setdefault(node, []).append(message)

    def send_message(self, message: AnyMessage, source_node: str, target_node: str):
        # In monolithic mode every node already reads the same shared log, and
        # callers record their response with add_self_message before sending it.
        # Appending again would duplicate the message in the shared transcript.
        if self.monolithic:
            return
        content = get_content(message)
        content = f'{source_node.upper()}: {content}'
        message.content = content
        message.additional_kwargs["source_node"] = source_node
        self.l1.setdefault(target_node, []).append(message)

    def compact_l1(self, compacted: AIMessage, node: str):
        self.l1[node] = [compacted]

    def format_messages(self, node: str, start_index: int = 0) -> str:
        messages = self.l1.get(node, [])
        messages = messages[max(start_index, 0):]
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

    def advance_l1_task_checkpoint(self):
        """Mark the end of the current task in the shared monolithic transcript."""
        if self.monolithic:
            self.l1_task_checkpoint = len(self.l1.get("all", []))

    def inject(self, node: str, task: dict[str, Any] | None = None) -> str:
        label = "Shared log" if self.monolithic else "Your log"
        l1 = f'''{label} (last is most recent):\n{self.format_messages(node)}'''
        if task is None:
            return l1

        return "\n\n".join(
            section
            for section in (
                self.inject_l2(task, node) if self.l2_enabled else "",
                self.inject_l3(task, node)
                if self.l3_enabled and node != "planner"
                else "",
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
        for key in ("target_files", "relevant_files"):
            value = task.get(key)
            if isinstance(value, str):
                files.append(value)
            elif isinstance(value, list):
                files.extend(item for item in value if isinstance(item, str))
        normalized = (
            normalize_workspace_path(path)
            for path in files
        )
        return list(dict.fromkeys(path for path in normalized if path is not None))

    def update_l2(self, path: str, summary: str):
        normalized = normalize_workspace_path(path)
        if normalized is not None and summary.strip():
            self.l2[normalized] = summary.strip()

    def track_programmer_file(self, path: str) -> bool:
        """Record a successfully mutated workspace-relative file for the next L2 pass."""
        normalized = normalize_workspace_path(path)
        if normalized is None:
            return False
        self.programmer_touched_files.add(normalized)
        return True

    def remove_l2(self, path: str):
        """Remove memory for a file that no longer exists."""
        normalized = normalize_workspace_path(path)
        if normalized is None:
            return
        self.l2.pop(normalized, None)
        self.l2_hashes.pop(normalized, None)
        self.programmer_touched_files.discard(normalized)

    def complete_l2_update(self, path: str, content_hash: str, summary: str):
        """Commit summary and hash atomically after a successful L2 model response."""
        normalized = normalize_workspace_path(path)
        summary = summary.strip()
        if normalized is None or not summary:
            return
        self.l2[normalized] = summary
        self.l2_hashes[normalized] = content_hash
        self.programmer_touched_files.discard(normalized)

    def mark_l2_unchanged(self, path: str):
        normalized = normalize_workspace_path(path)
        if normalized is not None:
            self.programmer_touched_files.discard(normalized)

    def valid_l2_summaries(self, paths: list[str]) -> list[tuple[str, str]]:
        """Return only summaries whose stored hash matches the current workspace file."""
        valid = []
        for candidate in paths:
            path = normalize_workspace_path(candidate)
            if path is None:
                continue
            summary = self.l2.get(path)
            if summary is None:
                continue

            try:
                content_hash = hashlib.sha256(tools.safe_path(path).read_bytes()).hexdigest()
            except (AssertionError, OSError):
                self.l2.pop(path, None)
                self.l2_hashes.pop(path, None)
                self.programmer_touched_files.discard(path)
                print(f"MEMORY INJECT L2 REMOVED UNREADABLE {path}")
                continue

            if self.l2_hashes.get(path) != content_hash:
                # Suppress stale (and legacy unhashed) entries immediately and
                # queue the file for the next L2 maintenance pass.
                self.programmer_touched_files.add(path)
                print(f"MEMORY INJECT L2 SUPPRESSED STALE {path}")
                continue

            valid.append((path, summary))
        return valid

    def inject_l2(self, task: dict[str, Any], node: str) -> str:
        if not self.l2_enabled:
            return ""
        if node == "planner":
            summaries = [
                f"{path}:\n{summary}"
                for path, summary in self.valid_l2_summaries(list(self.l2))
            ]
            if not summaries:
                return ""
            return (
                "Current workspace snapshot summaries. These briefly describe the relevant files' current state, "
                "but are not a declaration of implementation correctness:\n"
                + "\n\n".join(summaries)
            )

        files = self.relevant_files_for_task(task)
        summaries = [
            f"{path}:\n{summary}"
            for path, summary in self.valid_l2_summaries(files)
        ]
        if not summaries:
            return ""
        return (
            "Current workspace snapshot summaries. These briefly describe the relevant files' current state, "
            "but are not a declaration of implementation correctness:\n"
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

    def has_exact_l3_insight(self, insight: str) -> bool:
        normalized = " ".join(insight.lower().split())
        return any(
            " ".join(str(item.get("insight", "")).lower().split()) == normalized
            for item in self.l3
        )

    def insert_l3(
        self,
        abstract: str,
        insight: str,
        replace_ids: list[str] | None = None,
    ) -> bool:
        abstract = abstract.strip()
        insight = insight.strip()
        if not abstract or not insight:
            return False
        if self.has_exact_l3_insight(insight):
            return False
        embedding = self.embed_text(abstract)
        if not embedding:
            return False
        if replace_ids:
            replace_id_set = set(replace_ids)
            self.l3 = [
                item
                for item in self.l3
                if item.get("id") not in replace_id_set
            ]
        insight_id = f"l3-{self.l3_next_id}"
        self.l3_next_id += 1
        self.l3.append({
            "id": insight_id,
            "abstract": abstract,
            "insight": insight,
            "embedding": embedding,
        })
        return True

    def l3_reconciliation_view(self) -> list[dict[str, str]]:
        """Return the complete L3 list without embeddings for model reconciliation."""
        return [
            {
                "id": str(item.get("id", f"legacy-{index}")),
                "abstract": str(item.get("abstract", "")),
                "insight": str(item.get("insight", "")),
            }
            for index, item in enumerate(self.l3, start=1)
        ]

    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def retrieve_l3(self, query: str) -> list[dict[str, Any]]:
        if not query.strip() or not self.l3:
            return []
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return []
        scored = sorted(
            (
                (
                    self.cosine_similarity(query_embedding, item["embedding"]),
                    item,
                )
                for item in self.l3
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        score_log = ", ".join(
            f"{item.get('id', 'legacy')}={score:.4f}"
            for score, item in scored
        )
        print(
            "MEMORY INJECT L3 SCORES "
            f"(threshold={self.l3_similarity_threshold:.4f}): {score_log}"
        )
        return [
            item
            for score, item in scored
            if score >= self.l3_similarity_threshold
        ]

    def inject_l3(self, task: dict[str, Any], node: str) -> str:
        if not self.l3_enabled or node == "planner":
            return ""
        if not self.l3:
            print("MEMORY INJECT L3 SKIPPED (stored=0)")
            return ""
        query = "\n".join(
            str(task.get(key, ""))
            for key in (
                "task",
                "test_instructions",
                "target_files",
                "relevant_files",
            )
        )
        insights = self.retrieve_l3(query)
        if not insights:
            print(f"MEMORY INJECT L3 SKIPPED (stored={len(self.l3)}, retrieved=0)")
            return ""
        print(f"MEMORY INJECT L3 RETRIEVED {len(insights)} of {len(self.l3)}")
        lines = [
            f"- {item['insight']}"
            for item in insights
        ]
        return (
            "Historical project insights relevant to the current task. Use them as guidance, "
            "and verify them against current files when present behavior matters:\n"
            + "\n".join(lines)
        )
