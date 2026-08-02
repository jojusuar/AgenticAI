import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
import graph
import nodes
import tools
from memory import MyMemory, OllamaEmbeddingError


class L2InjectionTests(unittest.IsolatedAsyncioTestCase):
    def test_default_experiment_limits(self):
        self.assertEqual(0.75, graph.MONEY_LIMIT_DOLLARS)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.workspace_patch = patch.object(tools, "WORKSPACE", self.workspace)
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def store_summary(self, memory, path, content, summary="module summary"):
        file_path = self.workspace / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        memory.complete_l2_update(path, content_hash, summary)
        return file_path

    def test_json_parser_closes_a_missing_final_object_brace(self):
        parsed = nodes.parse_json_response('{"summary": "Recovered summary."')

        self.assertEqual({"summary": "Recovered summary."}, parsed)

    def test_json_parser_restores_missing_opening_object_prefix(self):
        parsed = nodes.parse_json_response(
            'insight": "Cross-module constraint."}'
        )

        self.assertEqual(
            {"insight": "Cross-module constraint."},
            parsed,
        )

    def test_planner_normalization_reads_legacy_top_level_task_fields(self):
        task, _, _ = nodes.normalize_planner_response(
            {
                "project_info": "Example",
                "finished": False,
                "task": {"task": "Implement feature"},
                "test_instructions": "Run tests",
                "target_files": ["feature.py"],
                "relevant_files": [],
                "target_file_structure": {},
            }
        )

        self.assertEqual(["feature.py"], task["target_files"])
        self.assertEqual("Run tests", task["test_instructions"])

    def test_all_model_clients_split_reasoning_from_content(self):
        for model in (
            nodes.planner_model,
            nodes.programmer_model,
            nodes.evaluator_model,
            nodes.compactor_model,
            nodes.memory_operator_model,
        ):
            self.assertEqual({"reasoning_split": True}, model.extra_body)

    async def test_node_heartbeat_refreshes_only_normal_completions(self):
        async def successful_node(state):
            return {"node_completed": True}

        async def failed_node(state):
            return {"timeout": True}

        with patch.object(graph.time, "monotonic", return_value=123.0):
            successful = await graph.with_node_heartbeat(successful_node)({})
            failed = await graph.with_node_heartbeat(failed_node)({})

        self.assertEqual(123.0, successful["last_node_completion_time"])
        self.assertFalse(successful["node_stall_timeout"])
        self.assertNotIn("last_node_completion_time", failed)

    def test_node_stall_timeout_uses_last_completion(self):
        state = {"last_node_completion_time": 100.0}

        with patch.object(
            graph.time,
            "monotonic",
            return_value=100.0 + graph.NODE_STALL_TIMEOUT_SECONDS,
        ):
            self.assertTrue(graph.node_stall_timeout_exceeded(state))

    def test_embedding_failure_has_explicit_termination_reason(self):
        reason = graph.determine_termination_reason(
            {},
            OllamaEmbeddingError("embedding service unavailable"),
        )

        self.assertEqual("embedding_failure", reason)

    async def test_model_call_aborts_when_input_usage_is_zero(self):
        class Model:
            ainvoke = AsyncMock(
                return_value=AIMessage(
                    content="{}",
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 1,
                        }
                    },
                )
            )

        with self.assertRaises(nodes.UsageReportingError):
            await nodes.invoke_model(Model(), "Prompt")

    def test_usage_reporting_failure_has_explicit_termination_reason(self):
        reason = graph.determine_termination_reason(
            {},
            nodes.UsageReportingError("missing usage"),
        )

        self.assertEqual("usage_reporting_failure", reason)

    async def test_compactor_receives_only_task_local_l1(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="L1_ONLY_CURRENT_TASK_MARKER"),
            "programmer",
        )
        self.store_summary(
            memory,
            "module.py",
            "VALUE = 1\n",
            summary="L2_SUMMARY_MARKER",
        )
        memory.l3 = [
            {
                "id": "l3-1",
                "abstract": "relevant current task",
                "insight": "L3_INSIGHT_MARKER",
                "embedding": [1.0],
            }
        ]
        memory.embed_text = lambda text: [1.0]
        response = AIMessage(
            content='{"summary": "Compacted task-local history."}'
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "relevant current task"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.compactor_node(state)

        prompt = invoke.await_args.args[1]
        self.assertIn("L1_ONLY_CURRENT_TASK_MARKER", prompt)
        self.assertNotIn("L2_SUMMARY_MARKER", prompt)
        self.assertNotIn("L3_INSIGHT_MARKER", prompt)
        self.assertLess(
            prompt.index("COMPACT EVERYTHING BELOW HERE:"),
            prompt.index("L1_ONLY_CURRENT_TASK_MARKER"),
        )
        self.assertIn("--- END SOURCE MATERIAL ---", prompt)
        self.assertEqual(
            "COMPACTED TASK HISTORY:\nCompacted task-local history.",
            memory.format_messages("programmer"),
        )

    async def test_compactor_retries_malformed_response_before_committing(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="not JSON",
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                        }
                    },
                ),
                AIMessage(
                    content='{"summary": "Recovered summary."}',
                    response_metadata={
                        "token_usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 3,
                        }
                    },
                ),
            ]
        )

        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.compactor_node(state)

        self.assertEqual(2, invoke.await_count)
        self.assertEqual(30, result["input_tokens"])
        self.assertEqual(5, result["output_tokens"])
        self.assertEqual(30, result["harness_input_tokens"])
        self.assertEqual(5, result["harness_output_tokens"])
        retry_prompt = invoke.await_args_list[1].args[1]
        self.assertTrue(
            retry_prompt.startswith("You were asked to return raw JSON matching")
        )
        self.assertIn("ORIGINAL_HISTORY", retry_prompt)
        self.assertEqual(
            "COMPACTED TASK HISTORY:\nRecovered summary.",
            memory.format_messages("programmer"),
        )

    async def test_compactor_truncates_history_after_two_unusable_responses(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(
            side_effect=[
                AIMessage(content="first garbage"),
                AIMessage(content="second garbage"),
            ]
        )

        with patch.object(nodes, "invoke_model", invoke):
            await nodes.compactor_node(state)

        self.assertEqual(2, invoke.await_count)
        self.assertEqual(
            "[history truncated]",
            memory.format_messages("programmer"),
        )

    async def test_compactor_truncates_history_when_retry_times_out(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(
            side_effect=[
                AIMessage(content="first garbage"),
                asyncio.TimeoutError(),
            ]
        )

        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.compactor_node(state)

        self.assertEqual(2, invoke.await_count)
        self.assertTrue(result["timeout"])
        self.assertEqual(
            "[history truncated]",
            memory.format_messages("programmer"),
        )

    async def test_compactor_truncates_history_when_retry_has_model_error(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(
            side_effect=[
                AIMessage(content="first garbage"),
                RuntimeError("retry failed"),
            ]
        )

        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.compactor_node(state)

        self.assertEqual(2, invoke.await_count)
        self.assertTrue(result["model_error"])
        self.assertEqual(
            "[history truncated]",
            memory.format_messages("programmer"),
        )

    async def test_compactor_truncates_immediately_after_initial_timeout(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.compactor_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertTrue(result["timeout"])
        self.assertEqual(
            "[history truncated]",
            memory.format_messages("programmer"),
        )

    async def test_compactor_truncates_immediately_after_initial_model_error(self):
        memory = MyMemory()
        memory.add_self_message(
            AIMessage(content="ORIGINAL_HISTORY"),
            "programmer",
        )
        state = {
            "memory": memory,
            "active_node": "programmer",
            "current_task": {"task": "Implement feature"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "money_limit": 1,
        }
        invoke = AsyncMock(side_effect=RuntimeError("initial failure"))

        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.compactor_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertTrue(result["model_error"])
        self.assertEqual(
            "[history truncated]",
            memory.format_messages("programmer"),
        )

    def test_matching_summary_is_orientation_not_correctness_claim(self):
        memory = MyMemory(l3_enabled=False)
        self.store_summary(memory, "src/example.py", "VALUE = 1\n")

        injected = memory.inject_l2(
            {"target_files": ["src/example.py"]}, "evaluator"
        )

        self.assertIn("module summary", injected)
        self.assertIn("Current workspace snapshot summaries", injected)
        self.assertIn("not a declaration of implementation correctness", injected)
        self.assertNotIn("L2", injected)
        self.assertNotIn("Prefer these summaries", injected)
        self.assertNotIn("completed tasks", injected)

    def test_equivalent_task_path_matches_normalized_l2_key(self):
        memory = MyMemory(l3_enabled=False)
        self.store_summary(memory, "src/example.py", "VALUE = 1\n")

        injected = memory.inject_l2(
            {"target_files": [r".\src\example.py"]}, "evaluator"
        )

        self.assertIn("src/example.py", injected)
        self.assertIn("module summary", injected)

    def test_changed_file_suppresses_summary_and_queues_refresh(self):
        memory = MyMemory(l3_enabled=False)
        file_path = self.store_summary(memory, "src/example.py", "VALUE = 1\n")
        file_path.write_text("VALUE = 2\n")

        injected = memory.inject_l2(
            {"target_files": ["src/example.py"]}, "programmer"
        )

        self.assertEqual("", injected)
        self.assertIn("src/example.py", memory.programmer_touched_files)

    def test_missing_file_removes_obsolete_summary(self):
        memory = MyMemory(l3_enabled=False)
        file_path = self.store_summary(memory, "src/example.py", "VALUE = 1\n")
        file_path.unlink()

        injected = memory.inject_l2({}, "planner")

        self.assertEqual("", injected)
        self.assertNotIn("src/example.py", memory.l2)
        self.assertNotIn("src/example.py", memory.l2_hashes)

    def test_planner_injection_includes_l2_but_excludes_l3(self):
        memory = MyMemory()
        self.store_summary(memory, "src/example.py", "VALUE = 1\n")
        memory.l3.append(
            {
                "abstract": "cross-module constraint",
                "insight": "Do not expose this historical insight to the planner.",
                "embedding": [1.0],
            }
        )

        injected = memory.inject("planner", {})

        self.assertIn("module summary", injected)
        self.assertNotIn("historical insight", injected)

    def test_l3_retrieval_returns_every_insight_above_threshold(self):
        memory = MyMemory(l2_enabled=False, l3_similarity_threshold=0.60)
        memory.embed_text = lambda text: [1.0, 0.0]
        memory.l3 = [
            {
                "id": "l3-1",
                "abstract": "one",
                "insight": "score one",
                "embedding": [1.0, 0.0],
            },
            {
                "id": "l3-2",
                "abstract": "point eight",
                "insight": "score point eight",
                "embedding": [0.8, 0.6],
            },
            {
                "id": "l3-3",
                "abstract": "point seven",
                "insight": "score point seven",
                "embedding": [0.7, 0.7141428429],
            },
            {
                "id": "l3-4",
                "abstract": "point five",
                "insight": "score point five",
                "embedding": [0.5, 0.8660254038],
            },
        ]

        injected = memory.inject_l3({"task": "query"}, "programmer")

        self.assertIn("score one", injected)
        self.assertIn("score point eight", injected)
        self.assertIn("score point seven", injected)
        self.assertNotIn("score point five", injected)
        self.assertIn("Historical project insights", injected)

    def test_l3_retrieval_query_includes_relevant_files(self):
        memory = MyMemory(l2_enabled=False)
        embedded_queries = []

        def embed(text):
            embedded_queries.append(text)
            return [1.0]

        memory.embed_text = embed
        memory.l3 = [
            {
                "id": "l3-1",
                "abstract": "shared transport constraint",
                "insight": "Keep the shared transport compatible.",
                "embedding": [1.0],
            }
        ]

        memory.inject_l3(
            {
                "task": "Update client",
                "test_instructions": "Run tests",
                "target_files": ["client.py"],
                "relevant_files": ["shared_transport.py"],
            },
            "programmer",
        )

        self.assertEqual(1, len(embedded_queries))
        self.assertIn("shared_transport.py", embedded_queries[0])

    async def test_l2_stops_after_call_that_reaches_budget(self):
        memory = MyMemory(l3_enabled=False)
        for path in ("a.py", "b.py"):
            (self.workspace / path).write_text(f"# {path}\n")
            memory.track_programmer_file(path)
        response = AIMessage(
            content='{"summary": "current file"}',
            response_metadata={
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 0}
            },
        )
        state = {
            "memory": memory,
            "current_task": {},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 0.0001,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)) as invoke:
            result = await nodes.l2_operator_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertEqual(100, result["input_tokens"])
        self.assertEqual(100, result["memory_input_tokens"])
        self.assertIn("a.py", memory.l2)
        self.assertIn("b.py", memory.programmer_touched_files)

    async def test_l2_prompt_is_not_conditioned_on_current_task(self):
        memory = MyMemory(l3_enabled=False)
        (self.workspace / "module.py").write_text("VALUE = 1\n")
        memory.track_programmer_file("module.py")
        response = AIMessage(content='{"summary": "Defines VALUE."}')
        state = {
            "memory": memory,
            "current_task": {"task": "TASK_ONLY_SENTINEL_DO_NOT_INCLUDE"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l2_operator_node(state)

        prompt = invoke.await_args.args[1]
        self.assertNotIn("TASK_ONLY_SENTINEL_DO_NOT_INCLUDE", prompt)
        self.assertNotIn("Current task:", prompt)
        self.assertIn("File path:\nmodule.py", prompt)
        self.assertIn("File content:\nVALUE = 1", prompt)

    async def test_l2_retries_malformed_summary_once_with_correction(self):
        memory = MyMemory(l3_enabled=False)
        (self.workspace / "module.py").write_text("VALUE = 1\n")
        memory.track_programmer_file("module.py")
        malformed = AIMessage(
            content="Defines VALUE without the requested JSON wrapper.",
            response_metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 1}
            },
        )
        valid = AIMessage(
            content='{"summary": "Defines VALUE."}',
            response_metadata={
                "token_usage": {"prompt_tokens": 12, "completion_tokens": 2}
            },
        )
        state = {
            "memory": memory,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(side_effect=[malformed, valid])
        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.l2_operator_node(state)

        self.assertEqual(2, invoke.await_count)
        retry_prompt = invoke.await_args_list[1].args[1]
        self.assertTrue(retry_prompt.startswith("You were asked to return raw JSON"))
        self.assertIn("but didn't", retry_prompt)
        self.assertEqual("Defines VALUE.", memory.l2["module.py"])
        self.assertNotIn("module.py", memory.programmer_touched_files)
        self.assertEqual(22, result["input_tokens"])
        self.assertEqual(3, result["output_tokens"])

    async def test_l2_does_not_retry_malformed_summary_after_budget_reached(self):
        memory = MyMemory(l3_enabled=False)
        (self.workspace / "module.py").write_text("VALUE = 1\n")
        memory.track_programmer_file("module.py")
        malformed = AIMessage(
            content="Not JSON.",
            response_metadata={
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 0}
            },
        )
        state = {
            "memory": memory,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 0.0001,
        }

        invoke = AsyncMock(return_value=malformed)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l2_operator_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertNotIn("module.py", memory.l2)
        self.assertIn("module.py", memory.programmer_touched_files)

    async def test_l3_skips_abstract_call_when_insight_reaches_budget(self):
        memory = MyMemory(l2_enabled=False)
        memory.embed_text = lambda text: [1.0]
        response = AIMessage(
            content='{"insights": ["Keep API and storage schemas aligned."]}',
            response_metadata={
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 0}
            },
        )
        state = {
            "memory": memory,
            "current_task": {"task": "completed"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 0.0001,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)) as invoke:
            result = await nodes.l3_operator_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertEqual(100, result["input_tokens"])
        self.assertEqual(1, len(memory.l3))
        self.assertEqual(memory.l3[0]["abstract"], memory.l3[0]["insight"])

    async def test_l3_retries_malformed_insight_once(self):
        memory = MyMemory(l2_enabled=False)
        memory.embed_text = lambda text: [1.0]
        malformed = AIMessage(content="not json")
        valid = AIMessage(
            content='{"insights": ["Keep API and storage schemas aligned."]}',
            response_metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 0}
            },
        )
        state = {
            "memory": memory,
            "current_task": {"task": "completed"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 0.00001,
        }

        invoke = AsyncMock(side_effect=[malformed, valid])
        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.l3_operator_node(state)

        self.assertEqual(2, invoke.await_count)
        self.assertEqual(10, result["input_tokens"])
        self.assertEqual(1, len(memory.l3))

    async def test_l3_does_not_retry_malformed_output_after_budget_reached(self):
        memory = MyMemory(l2_enabled=False)
        malformed = AIMessage(
            content="not json",
            response_metadata={
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 0}
            },
        )
        state = {
            "memory": memory,
            "current_task": {"task": "completed"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 0.00001,
        }

        invoke = AsyncMock(return_value=malformed)
        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.l3_operator_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertEqual(10, result["input_tokens"])
        self.assertEqual([], memory.l3)

    async def test_l3_slices_monolithic_log_from_previous_task_checkpoint(self):
        memory = MyMemory(l1_enabled=False, l2_enabled=False, l3_enabled=True)
        memory.add_self_message(AIMessage(content="old task message"), "programmer")
        memory.advance_l1_task_checkpoint()
        memory.add_self_message(AIMessage(content="current task message"), "programmer")
        response = AIMessage(content='{"insights": []}')
        state = {
            "memory": memory,
            "current_task": {"task": "current"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l3_operator_node(state)

        prompt = invoke.await_args.args[1]
        self.assertIn("current task message", prompt)
        self.assertNotIn("old task message", prompt)
        self.assertEqual(
            len(memory.l1.get("all", [])),
            memory.l1_task_checkpoint,
        )

    async def test_l3_receives_programmer_and_evaluator_task_logs(self):
        memory = MyMemory(l2_enabled=False)
        memory.add_self_message(
            AIMessage(content="PROGRAMMER_IMPLEMENTATION_EVIDENCE"),
            "programmer",
        )
        memory.add_self_message(
            AIMessage(content="EVALUATOR_PASS_EVIDENCE"),
            "evaluator",
        )
        response = AIMessage(content='{"insights": []}')
        state = {
            "memory": memory,
            "current_task": {"task": "completed"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l3_operator_node(state)

        prompt = invoke.await_args.args[1]
        self.assertIn("PROGRAMMER_IMPLEMENTATION_EVIDENCE", prompt)
        self.assertIn("EVALUATOR_PASS_EVIDENCE", prompt)
        self.assertIn("Programmer L1:", prompt)
        self.assertIn("Evaluator L1:", prompt)

    async def test_l3_reconciles_candidate_against_complete_existing_list(self):
        memory = MyMemory(l2_enabled=False)
        memory.embed_text = lambda text: [1.0]
        memory.insert_l3("Use legacy transport.", "All clients use the legacy transport.")
        memory.insert_l3("Preserve schema IDs.", "Schema IDs are stable across modules.")
        insight_response = AIMessage(
            content='{"insights":["All clients now use the unified transport."]}'
        )
        reconciliation_response = AIMessage(
            content=(
                '{"decisions":[{"candidate_index":0,"action":"REPLACE",'
                '"replace_ids":["l3-1"],'
                '"insight":"All clients use the unified transport.",'
                '"abstract":"Use the unified transport across clients."}]}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {"task": "migrate transport"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(
            side_effect=[insight_response, reconciliation_response]
        )
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l3_operator_node(state)

        reconciliation_prompt = invoke.await_args_list[1].args[1]
        self.assertIn('"id": "l3-1"', reconciliation_prompt)
        self.assertIn('"id": "l3-2"', reconciliation_prompt)
        self.assertIn("All clients use the legacy transport.", reconciliation_prompt)
        self.assertIn("Schema IDs are stable across modules.", reconciliation_prompt)
        self.assertEqual(2, len(memory.l3))
        self.assertFalse(
            any(item["id"] == "l3-1" for item in memory.l3)
        )
        self.assertTrue(
            any(item["id"] == "l3-2" for item in memory.l3)
        )
        self.assertTrue(
            any(
                item["insight"] == "All clients use the unified transport."
                for item in memory.l3
            )
        )

    async def test_l3_extracts_and_reconciles_up_to_three_candidates_in_two_calls(self):
        memory = MyMemory(l2_enabled=False)
        memory.embed_text = lambda text: [1.0]
        extraction_response = AIMessage(
            content=(
                '{"insights":['
                '"Keep transport adapters aligned across clients.",'
                '"Use stable schema identifiers across persistence modules.",'
                '"Validate migrations through the public loading boundary."'
                ']}'
            )
        )
        reconciliation_response = AIMessage(
            content=(
                '{"decisions":['
                '{"candidate_index":0,"action":"ADD","replace_ids":[],'
                '"insight":"Keep transport adapters aligned across clients.",'
                '"abstract":"Transport adapters stay aligned across clients."},'
                '{"candidate_index":1,"action":"DISCARD","replace_ids":[],'
                '"insight":"","abstract":""},'
                '{"candidate_index":2,"action":"ADD","replace_ids":[],'
                '"insight":"Validate migrations through the public loading boundary.",'
                '"abstract":"Migrations are validated through public loading."}'
                ']}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {"task": "complete integration work"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(
            side_effect=[extraction_response, reconciliation_response]
        )
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l3_operator_node(state)

        self.assertEqual(2, invoke.await_count)
        reconciliation_prompt = invoke.await_args_list[1].args[1]
        self.assertIn('"candidate_index": 0', reconciliation_prompt)
        self.assertIn('"candidate_index": 1', reconciliation_prompt)
        self.assertIn('"candidate_index": 2', reconciliation_prompt)
        self.assertEqual(2, len(memory.l3))
        self.assertTrue(
            any("transport adapters" in item["insight"] for item in memory.l3)
        )
        self.assertTrue(
            any("migrations" in item["insight"].lower() for item in memory.l3)
        )
        self.assertFalse(
            any("schema identifiers" in item["insight"] for item in memory.l3)
        )

    async def test_l3_discards_exact_duplicate_before_reconciliation_call(self):
        memory = MyMemory(l2_enabled=False)
        memory.embed_text = lambda text: [1.0]
        memory.insert_l3("Stable schema IDs.", "Schema IDs are stable across modules.")
        response = AIMessage(
            content='{"insights":["  schema ids ARE stable across modules.  "]}'
        )
        state = {
            "memory": memory,
            "current_task": {"task": "schema work"},
            "passed": True,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "input_rate_per_million": 1.0,
            "cached_input_rate_per_million": 1.0,
            "output_rate_per_million": 1.0,
            "money_limit": 1.0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            await nodes.l3_operator_node(state)

        self.assertEqual(1, invoke.await_count)
        self.assertEqual(1, len(memory.l3))

    async def test_abandoned_monolithic_task_advances_checkpoint_without_clearing_log(self):
        memory = MyMemory(l1_enabled=False, l2_enabled=False, l3_enabled=True)
        memory.add_self_message(AIMessage(content="abandoned attempt"), "programmer")
        state = {
            "memory": memory,
            "passed": False,
            "current_task": {"task": "abandoned"},
            "reflection_count": 3,
            "planner_handoff": {},
        }

        await nodes.task_cleanup_node(state)

        self.assertIn("abandoned attempt", memory.format_messages("planner"))
        self.assertEqual(
            len(memory.l1.get("all", [])),
            memory.l1_task_checkpoint,
        )

    async def test_planner_receives_and_consumes_previous_task_handoff(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        response = AIMessage(
            content=(
                '{"project_info":"Python project","finished":false,"task":{'
                '"task":"Add parser","test_instructions":"Run parser tests",'
                '"target_files":["parser.py"],"relevant_files":[],'
                '"target_file_structure":{"parser.py":[]}}}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {"task": "Previous work"},
            "planner_handoff": {
                "task": {"task": "Previous work"},
                "outcome": "ABANDONED",
                "evaluation_attempts": 3,
            },
            "planner_retries": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", invoke):
            result = await nodes.planner_node(state)

        prompt = invoke.await_args.args[1]
        self.assertIn("Immediate previous task outcome", prompt)
        self.assertIn('"outcome": "ABANDONED"', prompt)
        self.assertIn(
            "Do not create or expand test files as an implementation task",
            prompt,
        )
        self.assertIn(
            "validation procedures only in test_instructions",
            prompt,
        )
        self.assertIn(
            "Allow up to 3 target files",
            prompt,
        )
        self.assertIn(
            "Do not install, download, clone, inspect, import, or execute",
            prompt,
        )
        self.assertIn(
            "unrelated third-party",
            prompt,
        )
        self.assertEqual({}, result["planner_handoff"])
        self.assertEqual(1, result["tasks_attempted"])

    async def test_planner_rejects_more_than_three_target_files(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        response = AIMessage(
            content=(
                '{"project_info":"Python project","finished":false,"task":{'
                '"task":"Oversized task","test_instructions":"Run tests",'
                '"target_files":["a.py","b.py","c.py","d.py"],'
                '"relevant_files":[],"target_file_structure":{}}}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {},
            "planner_handoff": {},
            "planner_retries": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            result = await nodes.planner_node(state)

        self.assertFalse(result["node_completed"])
        self.assertEqual(1, result["planner_retries"])

    async def test_planner_reports_specific_schema_and_path_errors(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        response = AIMessage(
            content=(
                '{"project_info":"Python project","finished":false,"task":{'
                '"task":42,'
                '"target_files":["/tmp/absolute.py",7],'
                '"relevant_files":"src/context.py",'
                '"target_file_structure":[]}}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {},
            "planner_handoff": {},
            "planner_retries": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            result = await nodes.planner_node(state)

        feedback = memory.format_messages("planner")
        self.assertFalse(result["node_completed"])
        self.assertIn("`task.task` must be a string", feedback)
        self.assertIn("Missing required field `task.test_instructions`", feedback)
        self.assertIn("`task.target_files[0]` is absolute", feedback)
        self.assertIn("`task.target_files[1]` must be a string", feedback)
        self.assertIn("`task.relevant_files` must be a list", feedback)
        self.assertIn(
            "`task.target_file_structure` must be an object",
            feedback,
        )

    async def test_planner_normalizes_task_paths_before_accepting_task(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        response = AIMessage(
            content=(
                '{"project_info":"Python project","finished":false,"task":{'
                '"task":"Add parser","test_instructions":"Run parser tests",'
                '"target_files":["./src\\\\parser.py"],'
                '"relevant_files":["./src\\\\types.py"],'
                '"target_file_structure":{}}}'
            )
        )
        state = {
            "memory": memory,
            "current_task": {},
            "planner_handoff": {},
            "planner_retries": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            result = await nodes.planner_node(state)

        self.assertTrue(result["node_completed"])
        self.assertEqual(["src/parser.py"], result["current_task"]["target_files"])
        self.assertEqual(["src/types.py"], result["current_task"]["relevant_files"])

    async def test_planner_rejects_unsafe_and_canonically_duplicate_paths(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        responses = [
            AIMessage(
                content=(
                    '{"project_info":"Python project","finished":false,"task":{'
                    '"task":"Escape workspace","test_instructions":"None",'
                    '"target_files":["../escape.py"],"relevant_files":[],'
                    '"target_file_structure":{}}}'
                )
            ),
            AIMessage(
                content=(
                    '{"project_info":"Python project","finished":false,"task":{'
                    '"task":"Duplicate paths","test_instructions":"None",'
                    '"target_files":["src/a.py","./src/a.py"],"relevant_files":[],'
                    '"target_file_structure":{}}}'
                )
            ),
        ]
        state = {
            "memory": memory,
            "current_task": {},
            "planner_handoff": {},
            "planner_retries": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        for response in responses:
            with self.subTest(response=response.content):
                with patch.object(
                    nodes,
                    "invoke_model",
                    AsyncMock(return_value=response),
                ):
                    result = await nodes.planner_node(state)
                self.assertFalse(result["node_completed"])

    async def test_planner_handoff_counts_every_completed_evaluation(self):
        cases = [
            ("PASS", 2, 3, 2),
            ("FAIL", 0, 1, 1),
        ]
        for status, prior_failures, expected_attempts, expected_reflections in cases:
            with self.subTest(status=status, prior_failures=prior_failures):
                memory = MyMemory(l2_enabled=False, l3_enabled=False)
                response = AIMessage(
                    content=(
                        f'{{"status":"{status}","stacktrace":"",'
                        '"reason":"test result"}'
                    )
                )
                state = {
                    "memory": memory,
                    "project_info": "",
                    "current_task": {"task": "Evaluate change"},
                    "planner_handoff": {},
                    "reflection_count": prior_failures,
                    "tasks_completed": 4,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                }

                invoke = AsyncMock(return_value=response)
                with patch.object(nodes, "invoke_model", invoke):
                    result = await nodes.evaluator_node(state)

                prompt = invoke.await_args.args[1]
                self.assertIn(
                    "Do not install, download, clone, inspect, import, or execute",
                    prompt,
                )
                self.assertTrue(result["node_completed"])
                self.assertEqual(
                    expected_attempts,
                    result["planner_handoff"]["evaluation_attempts"],
                )
                self.assertEqual(
                    expected_reflections,
                    result["reflection_count"],
                )
                self.assertEqual(
                    5 if status == "PASS" else 4,
                    result["tasks_completed"],
                )

    async def test_task_cleanup_marks_failed_handoff_abandoned(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        memory.l1["programmer"] = [AIMessage(content="attempt")]
        state = {
            "memory": memory,
            "passed": False,
            "current_task": {"task": "Previous work"},
            "reflection_count": 3,
            "planner_handoff": {
                "task": {"task": "Previous work"},
                "outcome": "FAILED",
                "evaluation_attempts": 3,
                "final_evaluator_feedback": {"reason": "tests failed"},
            },
        }

        result = await nodes.task_cleanup_node(state)

        self.assertEqual({}, memory.l1)
        self.assertEqual("ABANDONED", result["planner_handoff"]["outcome"])
        self.assertNotIn("tasks_completed", result)
        self.assertEqual(
            "tests failed",
            result["planner_handoff"]["final_evaluator_feedback"]["reason"],
        )

    async def test_programmer_reported_files_are_used_only_when_l2_enabled(self):
        response = AIMessage(
            content='{"status":"DONE","touched_files":["src/generated.py","src/generated.py"]}',
            response_metadata={
                "token_usage": {"prompt_tokens": 12, "completion_tokens": 3}
            },
        )
        base_state = {
            "project_info": "",
            "current_task": {"task": "Generate file"},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        l2_memory = MyMemory(l3_enabled=False)
        l2_invoke = AsyncMock(return_value=response)
        with patch.object(nodes, "invoke_model", l2_invoke):
            l2_result = await nodes.programmer_node(
                {**base_state, "memory": l2_memory}
            )

        baseline_memory = MyMemory(memory=False)
        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            baseline_result = await nodes.programmer_node(
                {**base_state, "memory": baseline_memory}
            )

        self.assertTrue(l2_result["node_completed"])
        self.assertEqual(12, l2_result["harness_input_tokens"])
        self.assertEqual(3, l2_result["harness_output_tokens"])
        self.assertEqual({"src/generated.py"}, l2_memory.programmer_touched_files)
        prompt = l2_invoke.await_args.args[1]
        self.assertIn("authored source file", prompt)
        self.assertIn("*.egg-info", prompt)
        self.assertIn("Do not report generated artifacts", prompt)
        self.assertIn(
            "Do not install, download, clone, inspect, import, or execute",
            prompt,
        )
        self.assertTrue(baseline_result["node_completed"])
        self.assertEqual(12, baseline_result["harness_input_tokens"])
        self.assertEqual(3, baseline_result["harness_output_tokens"])
        self.assertEqual(set(), baseline_memory.programmer_touched_files)

    async def test_programmer_done_requires_valid_touched_files(self):
        memory = MyMemory(l3_enabled=False)
        response = AIMessage(content='{"status":"DONE"}')
        state = {
            "memory": memory,
            "project_info": "",
            "current_task": {},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            result = await nodes.programmer_node(state)

        self.assertFalse(result["node_completed"])
        self.assertIn(
            "Missing required field `touched_files`",
            memory.format_messages("programmer"),
        )

    async def test_evaluator_reports_specific_schema_errors(self):
        memory = MyMemory(l2_enabled=False, l3_enabled=False)
        response = AIMessage(content='{"status":"MAYBE","reason":[]}')
        state = {
            "memory": memory,
            "project_info": "",
            "current_task": {"task": "Evaluate change"},
            "reflection_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock(return_value=response)):
            result = await nodes.evaluator_node(state)

        feedback = memory.format_messages("evaluator")
        self.assertFalse(result["node_completed"])
        self.assertIn("`status` must be `PASS` or `FAIL`", feedback)
        self.assertIn("Missing required field `stacktrace`", feedback)
        self.assertIn("`reason` must be a string", feedback)

    async def test_reported_deleted_file_removes_l2_entry(self):
        memory = MyMemory(l3_enabled=False)
        memory.l2["deleted.py"] = "obsolete"
        memory.l2_hashes["deleted.py"] = "old-hash"
        memory.track_programmer_file("deleted.py")
        state = {
            "memory": memory,
            "current_task": {},
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        await nodes.l2_operator_node(state)

        self.assertNotIn("deleted.py", memory.l2)
        self.assertNotIn("deleted.py", memory.l2_hashes)
        self.assertNotIn("deleted.py", memory.programmer_touched_files)

    async def test_unsupported_l2_paths_are_removed_instead_of_retried(self):
        memory = MyMemory(l3_enabled=False)
        (self.workspace / "binary.dat").write_bytes(b"\xff\xfe\x00")
        (self.workspace / "generated").mkdir()
        for path in ("binary.dat", "generated"):
            memory.l2[path] = "obsolete summary"
            memory.l2_hashes[path] = "obsolete hash"
            memory.track_programmer_file(path)
        state = {
            "memory": memory,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

        with patch.object(nodes, "invoke_model", AsyncMock()) as invoke:
            await nodes.l2_operator_node(state)

        invoke.assert_not_awaited()
        self.assertEqual(set(), memory.programmer_touched_files)
        self.assertNotIn("binary.dat", memory.l2)
        self.assertNotIn("generated", memory.l2)


if __name__ == "__main__":
    unittest.main()
