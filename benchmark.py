import json
import time
import asyncio
import graph
from human_eval.data import read_problems

OUTPUT_JSONL = "samples.jsonl"
METRICS_JSON = "metrics.json"

problems = read_problems()
samples = []
metrics = []

agent = graph.agent

async def run_benchmark():

    samples = []
    metrics = []

    for i, (task_id, problem) in enumerate(problems.items()):

        print(f"\nRunning {task_id}")

        start_time = time.time()

        result = await agent.ainvoke({
            "messages": [],
            "prompt": problem["prompt"],
            "canonical_solution": problem["canonical_solution"],
            "entrypoint": problem['entry_point'],
            "test": problem['test'],
            "completion": "",
            "reflection_count": 0,
            "passed": False,
            "start_time": start_time,
            "elapsed_time": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "fixing": False,
            "task_id": task_id,
        })

        elapsed = time.time() - start_time

        completion = result["completion"]

        samples.append({
            "task_id": task_id,
            "completion": completion
        })

        metrics.append({
            "task_id": task_id,
            "passed": result["passed"],
            "reflection_loops": result["reflection_count"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": (
                result["input_tokens"]
                + result["output_tokens"]
            ),
            "elapsed_seconds": elapsed,
            "completion_length": len(completion),
        })

        print(f"Passed: {result['passed']}")
        print(f"Loops: {result['reflection_count']}")
        print(f"Token usage:  Input: {result['input_tokens']}  Output: {result['output_tokens']}")
        print(f"Time: {elapsed:.2f}s")

    return samples, metrics

samples, metrics = asyncio.run(run_benchmark())


with open(OUTPUT_JSONL, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

total = len(metrics)
passed = sum(1 for m in metrics if m["passed"])
avg_tokens = sum(m["total_tokens"] for m in metrics) / total
avg_time = sum(m["elapsed_seconds"] for m in metrics) / total
avg_loops = sum(m["reflection_loops"] for m in metrics) / total

print("\n==============================")
print("BENCHMARK SUMMARY")
print("==============================")
print(f"Tasks: {total}")
print(f"Evaluator Passes: {passed}/{total}")
print(f"Average Tokens: {avg_tokens:.2f}")
print(f"Average Time: {avg_time:.2f}s")
print(f"Average Reflection Loops: {avg_loops:.2f}")

print(f"\nSaved completions to {OUTPUT_JSONL}")
print(f"Saved metrics to {METRICS_JSON}")