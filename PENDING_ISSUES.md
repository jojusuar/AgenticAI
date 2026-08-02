# Pending Issues Affecting the Memory Condition

This document records unresolved implementation and experimental-design issues that
could make the composed-memory condition perform worse than the monolithic baseline
for reasons other than the usefulness of memory itself. These issues should be
resolved, controlled, or disclosed before collecting the final experimental results.

## 1. Resolved: L2 memory is committed before evaluation

**Current behavior**

The successful programmer route is:

```text
programmer DONE -> L2 operator -> evaluator
```

The L2 operator hashes and summarizes every tracked file as soon as the programmer
reports completion. At that point, the implementation has not passed its task tests.

**Resolution**

The ordering is intentional. L2 is a descriptive snapshot of the current files, not
a correctness judgment. Refreshing it before evaluation prevents the evaluator from
receiving summaries of the previous file state. Evaluator feedback remains the
separate source of correctness information.

Injection no longer calls L2 "memory from completed tasks" or instructs agents to
prefer it over files. It explicitly presents L2 as orientation rather than evidence
of correctness and tells agents to consult files for exact or current behavior.

## 2. Resolved: Memory maintenance checks the monetary budget between calls

**Current behavior**

The graph checks the monetary limit between graph nodes. However:

- L2 can make one model call for every touched file inside a single node execution.
- L3 can make one call to extract an insight and another to produce its retrieval
  abstract.
- Neither operator checks the remaining budget before each internal call.

**Why this can hurt the memory condition**

The memory condition may spend its remaining allowance on several maintenance calls
and then terminate before evaluating or continuing implementation. The baseline has
no corresponding L2/L3 calls and can allocate the same nominal allowance to planning,
programming, and evaluation.

Some maintenance cost is a legitimate part of the memory treatment. Multiple calls
made after the budget has already been exhausted are not: they create uncontrolled
overshoot and can cause unequal realized expenditure across conditions.

**Resolution**

Each memory-operator response is added to the run's accumulated usage immediately.
L2 stops before processing another file when that total reaches the monetary limit.
L3 stops before abstract generation when its insight call reaches the limit; a valid
insight is retained using itself as retrieval text. The graph's existing routing
then terminates the run from the updated accumulated totals.

One model call can still cross the threshold because its final usage is not known in
advance. The final statistics record that realized overshoot.

## 3. Resolved: Task cleanup preserves immediate planner continuity

**Current behavior**

After L3 maintenance, `clear_all_l1()` removes every separate L1 log, including the
planner's history. The graph retains `project_info` in state, but the planner prompt
does not include the previous `project_info` value.

The next planner invocation therefore receives the specification, L2/L3 memory, and
an empty planner log.

**Why this can hurt the memory condition**

The planner loses prior architectural reasoning, task sequencing, assumptions, and
knowledge that did not qualify for L2 or L3 storage. The monolithic baseline retains
the complete planning history.

This is especially damaging when:

- An important fact concerns files that were not modified.
- A file was changed through Bash and never entered L2.
- An architectural observation was useful but did not satisfy the deliberately
  strict L3 criteria.
- The next task depends on why an earlier task was selected, not merely on the code
  that task produced.

The comparison can consequently measure destructive planner amnesia rather than the
quality of composed memory.

**Resolution**

Task-local L1 is still cleared so programmer and evaluator history from completed
tasks does not compete with L2/L3 context. The evaluator instead records a
deterministic `planner_handoff` in graph state containing the immediate previous
task, its outcome, evaluation-attempt count, and final evaluator feedback on
failure. Cleanup changes a final failed outcome to `ABANDONED`.

The next planner invocation receives this handoff independently of L1 and consumes
it after producing a valid plan. Planner retries retain it. This policy applies to
the composed-memory condition; the monolithic baseline already retains its shared
history.

## 4. Resolved: L2 tracking combines tool observation and programmer reporting

**Current behavior**

The harness records a touched file only after successful `write_file` or
`str_replace` calls. The programmer can also modify the workspace through Bash using
formatters, generators, package managers, `sed`, shell redirection, or other commands.
Those mutations are not tracked.

**Why this can hurt the memory condition**

Files modified through Bash may never receive an L2 summary. If a summary already
exists, a later Bash mutation may leave it stale. The composed condition then loses
or misrepresents implementation knowledge that remains visible in the monolithic
transcript.

This also makes memory quality depend on an incidental model choice— which editing
tool it selected—rather than on the generated code.

**Resolution**

Every programmer completion response, in every experimental condition, must declare
all authored source files created, modified, moved, or deleted during the task,
including indirect source changes made through Bash and other programs. Authored
source includes code, tests, manifests, configuration, and maintained documentation,
but excludes caches, build output, coverage data, and generated metadata such as
`*.egg-info`. L2 uses the union of these declared paths and paths observed through
successful dedicated write tools. The baseline validates the same response schema
but does not use the reported list.

Reported paths are normalized, constrained to workspace-relative paths, and
deduplicated by the tracking set. Reported deletions remove obsolete L2 entries.
This policy is tool-independent but model-reported: an omitted path cannot be
recovered unless a dedicated write tool observed it.

## 5. Resolved: Stored L2 summaries are validated when injected

**Current behavior**

L2 stores a content hash when a summary is created, but injection does not verify that
the current file still matches that hash.

**Why this can hurt the memory condition**

An undetected external or Bash-based modification can leave a stale summary in L2.
Agents are then encouraged to prefer that summary over reading the current file.
Stale memory is therefore not merely missing context; it can actively displace
correct context.

**Resolution**

Before injection, each candidate file is hashed and compared with the hash stored
when its summary was produced. A mismatched or unhashed summary is suppressed and
the file is queued for the next L2 refresh. Summaries for missing or unreadable files
are removed. The revised injection wording describes valid summaries as orientation
rather than a correctness source.

## 6. Resolved: Planner receives current workspace summaries but not L3 history

**Previous behavior**

The planner received every stored L2 summary and every stored L3 insight. L3
semantic top-k retrieval was used for programmer and evaluator contexts, but not
for the planner.

**Resolution**

The planner receives every hash-valid L2 summary because planning the next task
requires a compact view of the whole current workspace. This growth follows the
repository's current structure and is intentional.

L3 is no longer injected into the planner. Historical cross-task insights can grow
independently of the current workspace and lack a concrete retrieval query before a
task has been selected. Programmer and evaluator continue to receive top-k L3
retrieval after the planner provides a concrete task. Planner L1 remains available
for retries, and immediate cross-task continuity comes from `planner_handoff`.

## 7. Partially mitigated: L1 is cleared even when L3 maintenance fails

**Current behavior**

After the L3 operator finishes its exception-handling path, it clears all separate L1
logs. This can happen when:

- The L3 response is malformed.
- The model call times out.
- A nonterminal provider/model error occurs.
- Abstract generation fails after a valid insight was produced.

A valid response stating that no durable insight exists is different from a failed
maintenance attempt, but both currently lead to cleanup.

**Why this can hurt the memory condition**

The treatment can lose its detailed task history without successfully replacing any
of it with durable memory. The baseline retains its transcript. This creates
information loss caused by operator reliability rather than by the memory policy.

**Resolution**

L3 gets one retry per maintenance run when either the insight or abstract response
is malformed. The retry is skipped if the malformed call already reached the
monetary limit. Timeouts and provider/model failures are not retried.

Task-local L1 is still cleared if that retry also produces malformed output or if
another maintenance failure occurs. Those cases are accepted as unlikely; retaining
L1 for further attempts would add model-call and token cost that is not justified by
their expected frequency.

## 8. Resolved: L3 receives only the current monolithic task slice

**Current behavior**

When L1 is disabled but L3 is enabled, `format_messages("programmer")` resolves to
the complete shared append-only log. The prompt now labels this accurately, but the
underlying context still contains all earlier tasks and all agents.

**Why this can hurt the memory condition**

The operator is asked to extract an insight from the task that just passed, but old
tasks can dominate the supplied context. It may:

- Re-extract an earlier insight.
- Attribute an old failure mode to the current task.
- Produce duplicates.
- Spend progressively more tokens on historical context.

Prompt wording reduces ambiguity but does not isolate the evidence.

**Resolution**

Memory maintains a checkpoint index into the shared message list. L3 formats only
the messages between the previous task-end checkpoint and the current end of the
log, while normal monolithic agent context continues to use the complete transcript.
After L3 maintenance, the checkpoint advances to the current list length.

Abandoned monolithic tasks also pass through task cleanup, which advances the
checkpoint without clearing the shared transcript. Their messages therefore remain
available to the baseline agents but are excluded from a later successful task's L3
maintenance input.

## 9. Ignored: L2 summarization cost scales with full file size

**Current behavior**

For every changed text file, the L2 operator sends the complete current file to the
memory model. There is no file-size threshold, structural extraction, or incremental
summary update.

**Why this can hurt the memory condition**

A small change to a large file can consume a disproportionate amount of the monetary
budget. A task touching several large files can spend more on memory maintenance than
on implementation or evaluation. This cost pattern does not exist in the baseline.

Although maintenance cost belongs in an equal-cost treatment, avoidable full-file
reprocessing tests an inefficient L2 implementation rather than the value of module
memory.

**Accepted design cost**

No remediation will be implemented. L2 promises a concise description of each
file's current state, and the complete current file is the least lossy source for
that description. Updating a previous summary from a diff can preserve old
omissions, lose unchanged context needed to interpret a change, and accumulate drift
across tasks. Periodic full refresh would add an arbitrary policy while retaining
much of the same cost.

Unchanged files are already skipped by content hash, only reported or tool-observed
changed files are processed, and internal budget checks stop additional maintenance
calls after the monetary limit is reached. Full-file processing cost for changed
files is therefore treated as an inherent, disclosed cost of the L2 design.

## 10. Resolved: Usage separates harness work from memory overhead

**Current behavior**

The harness records total input, cached input, output, and reasoning tokens, but does
not attribute usage to planner, programmer, evaluator, compactor, L2, or L3.

**Why this matters**

The monetary limit is still enforceable, but a worse memory result cannot be
diagnosed. It will be unclear whether the condition failed because memory content was
harmful or because maintenance consumed the budget before productive work.

This does not directly lower performance, but it prevents the experiment from
distinguishing a conceptual failure of composed memory from an inefficient
implementation.

**Resolution**

Global token totals remain the source for budget enforcement. Parallel state
counters now attribute planner, programmer, evaluator, and compactor calls to core
`harness_usage`, while L2 and L3 model calls accumulate under `memory_overhead`.
Both categories track input, cached input, output, and reasoning tokens.

Final `usage.json` reports both category token totals and their calculated monetary
cost alongside the existing run totals. Local embedding calls do not report provider
tokens and are therefore outside these token counters.

## 11. Resolved: L3 retrieval uses reconciliation and a relevance threshold

**Current behavior**

When L3 contains entries, retrieval ranks every insight by cosine similarity and
returns the first `k`. There is no minimum relevance score, duplicate suppression,
or invalidation policy for insights made obsolete by later implementation changes.

**Why this can hurt the memory condition**

Once at least `k` insights exist, programmer and evaluator receive `k` entries even
when none is meaningfully related to the current task. Duplicate insights can crowd
out distinct ones, and old architectural guidance can conflict with the current
hash-validated L2 workspace description. The baseline has no privileged retrieval
section that labels these results as relevant.

**Implemented mitigation**

Exact duplicates are suppressed deterministically. For every non-duplicate
candidate, the existing second L3 call now receives the complete stored insight list
with stable IDs and returns one bounded `ADD`, `REPLACE`, or `DISCARD` mutation. It
does not rewrite the list. `REPLACE` can retire multiple directly contradicted or
superseded entries even when they are not close embedding neighbors. Only added or
replacement entries are embedded, using the local embedding service.

Malformed reconciliation output uses the existing single L3 retry allowance. When
the monetary limit prevents reconciliation, the valid candidate is retained without
the second call.

**Retrieval resolution**

Top-k retrieval has been removed. Every stored insight whose cosine similarity is at
least the configured threshold is injected; no insight below the threshold is
included, and there is no count cap on qualifying results. The initial fixed
threshold is `0.60`, parameterized in `graph.py` and accepted through
`l3_similarity_threshold=` or `--l3-similarity-threshold=`.

Retrieval logs every candidate score and the active threshold. Injected L3 is labeled
as historical project guidance that must be verified against current files when
present behavior matters. The threshold must remain fixed during final experiments
and is recorded in `usage.json`.

## 12. Resolved: Node-completion heartbeat replaces workspace-idle tracking

**Current behavior**

The workspace-change timestamp is updated only after successful `write_file` or
`str_replace` tool calls. A programmer can modify files through Bash and accurately
report them in `touched_files`, but that completion does not reset the timestamp.
Time spent in L2 and L3 also continues to count toward workspace idleness.

**Why this can hurt the memory condition**

Runs that use Bash for legitimate mutations can terminate as idle despite making
progress. The composed-memory condition then spends additional wall-clock time in
L2/L3 maintenance, making it more likely than the baseline to cross the same idle
threshold after an equally recent implementation change.

**Resolution**

Filesystem mutation is no longer used as the liveness signal. State now maintains
`last_node_completion_time`, refreshed after normal completion of planner,
programmer, evaluator, tool, L2, L3, and task-cleanup nodes. The 30-minute
`node_stall_timeout` therefore measures graph traversal rather than source-file
activity and treats Bash, testing, and memory maintenance consistently.

Timeouts, model errors, rate limits, and authentication failures do not refresh the
heartbeat. Compactor responses are also deliberately excluded because compaction is
a failure handler; repeated timeout-to-compaction loops must age toward termination
instead of perpetually resetting liveness. Individual model and Bash calls retain
their shorter call-level timeouts.

## 13. Resolved: Embedding failures terminate runs with an explicit cause

**Current behavior**

L3 retrieval embeds its query while programmer and evaluator prompts are being
constructed. Prompt construction occurs before those nodes enter their exception
handling. L3 insertion also deliberately re-raises embedding failures. The startup
preflight verifies availability only once.

**Why this can hurt the memory condition**

A transient local embedding-service failure can terminate a memory-enabled run for
an infrastructure reason that does not exist in the baseline. If counted as task
failure, this lowers measured memory performance independently of memory quality.

**Resolution**

Embedding failures remain terminal so a run never silently degrades into a different
memory condition. `OllamaEmbeddingError` is classified before the generic run-error
path and recorded in `usage.json` with
`termination_reason: "embedding_failure"` and `embedding_failed: true`.

Experiment analysis should treat this as an infrastructure termination rather than
an implementation or memory-quality failure.

## 14. Ignored: L1 formatting copies complete tool arguments

**Current behavior**

`format_messages()` renders every tool argument with `repr()`. For `write_file`,
`str_replace`, Bash, and similar calls, this can copy large source payloads or
commands into agent context. L3 then consumes the formatted programmer L1.

**Why this can hurt the memory condition**

Large tool arguments duplicate file content already represented by L2, increase core
prompt cost, and can dominate the evidence given to L3. This encourages L3 to
re-summarize implementation detail rather than extract cross-module knowledge.

**Accepted design cost**

No remediation will be implemented. In the current architecture, programmer
behavior is often expressed primarily through `write_file`, `str_replace`, Bash, and
other tool arguments, while tool results may contain only a success marker or byte
count. Removing payloads would deprive L3 of the implementation evidence needed to
identify cross-module interfaces, constraints, repairs, and integration decisions.

Replacing this evidence would require a broader redesign, such as providing L3 with
updated L2 summaries plus a separate structured change log, which could itself lose
relationships present in the actual edits. Complete tool arguments are therefore
retained as an evidence-rich input cost. Their model-token impact remains visible in
the harness and memory-overhead accounting and should be revisited only if prompt
size becomes a demonstrated failure mode.

## 15. Task paths are not normalized before L2 lookup — resolved

**Resolution**

A shared workspace-relative path normalizer now canonicalizes programmer
declarations, write-tool tracking, planner `target_files` and `relevant_files`, L2
storage, and L2 retrieval. Equivalent dot-segment and backslash variants resolve to
the same POSIX-style key. Absolute, drive-prefixed, traversal, empty, and null-byte
paths are rejected. Planner tasks are validated after normalization, so canonical
duplicates and target/relevant overlap are also rejected at the schema boundary.

## 16. Planner handoff reports failure count as evaluation-attempt count — resolved

**Resolution**

Every well-formed PASS or FAIL now records `reflection_count + 1` as the completed
evaluation-attempt count. This preserves `reflection_count` as a failure/retry
counter while accurately reporting immediate passes and passes after earlier
failures to the planner.

## 17. L2 summarization is conditioned on the current task — resolved

**Resolution**

The L2 operator no longer reads or injects `current_task`. Its prompt contains only
the task-independent summary contract, normalized file path, and complete current
file content, avoiding both repeated task-token cost and task-conditioned summary
bias.

## 18. Compaction re-ingests durable memory into L1 — resolved

**Resolution**

The compactor now receives only the active agent's task-local formatted L1 log.
`current_task` remains separate context for deciding what to preserve, but L2 and L3
are no longer included in the material being compacted. This prevents durable
memory from being duplicated or frozen into an L1 summary that can outlive later
workspace changes.

## 19. Separate-memory L3 omits successful evaluator evidence — resolved

**Resolution**

L3 extraction in the separate-memory condition now receives labeled programmer and
evaluator task-local logs. Successful evaluator tool calls, results, and final PASS
response are therefore available for durable testing or integration insights, while
the monolithic condition continues to use its all-agent current-task checkpoint
slice.

## 20. L3 retrieval queries omit relevant files — resolved

**Resolution**

Semantic retrieval queries now include planner-declared `relevant_files` alongside
the task, test instructions, and target files. This gives integration-boundary and
dependency insights the same filename-level retrieval signal as directly modified
files.

## 21. Unsupported L2 paths remain permanently queued — resolved

**Resolution**

Deterministically unsupported entries are no longer retried on every later task.
Missing paths, directories and other non-files, and non-UTF-8 files remove any
obsolete L2 entry and leave the touched-file queue. Other `OSError` failures remain
queued because they may be transient.

## 22. Malformed L2 summaries wait until a later task for retry — resolved

**Resolution**

Each file now gets one immediate retry when its first L2 response does not contain a
valid string `summary`. The retry prompt begins with a brief correction stating that
the required raw JSON object was not returned, then repeats the original
task-independent file-summary prompt. Both malformed payloads are logged. The retry
is skipped when the first call has already reached the monetary limit, and a second
malformed result leaves the file queued for a later task cycle.

## 23. Think-tag cleanup can corrupt content and L1 history — resolved

**Resolution**

Raw-response logging showed that the model can begin its JSON object immediately
before `</think>` and finish it afterward. Removing the complete think block then
left only the suffix and made a structurally valid intended response unparseable.
Think-only or orphan-tag responses could also enter L1 and make compaction appear
to erase a turn.

Every MiniMax client now requests `reasoning_split=true`, which keeps model
reasoning in the response's reasoning metadata and puts only the final answer in
`content`. Planner, programmer, evaluator, compactor, L2, and L3 consume that clean
content directly. The regex think-tag remover and think-boundary JSON
reconstruction have been removed.

The lenient JSON parser also now closes an otherwise valid object that is missing
only its final brace. Responses that still fail schema validation continue through
their existing retry policy.

## Recommended resolution order

Before final data collection, prioritize:

1. ~~Move L2 maintenance after evaluator PASS.~~ Resolved by defining L2 as a
   hash-validated current-workspace snapshot and removing correctness implications
   from injection wording.
2. ~~Enforce the monetary budget inside L2 and L3.~~ Implemented.
3. ~~Preserve planner continuity.~~ Implemented with a one-cycle task-outcome
   handoff.
4. ~~Detect filesystem mutations independently of editing tool.~~ Implemented with
   mandatory programmer reporting unioned with observed write-tool mutations.
5. ~~Validate L2 hashes at injection time.~~ Implemented.
6. ~~Avoid clearing L1 after failed L3 maintenance.~~ Partially mitigated with one
   malformed-output retry; further retry cost is not justified.
7. ~~Bound planner memory injection.~~ Resolved by excluding L3 while intentionally
   retaining the complete hash-valid L2 workspace view.
8. ~~Slice monolithic history by task for L3.~~ Implemented with a shared-log
   task-end checkpoint.
9. ~~Reduce full-file L2 reprocessing.~~ Ignored as an accepted design cost; summary
   plus diff is too lossy for a current-file description.
10. ~~Add usage attribution.~~ Implemented as separate core harness and memory-node
    overhead counters.
11. ~~Replace workspace-idle tracking with a node-completion heartbeat.~~ Implemented;
    compaction deliberately does not refresh it.
12. ~~Define a deterministic embedding-failure policy.~~ Embedding failures remain
    terminal and are recorded with an explicit termination cause.
13. ~~Normalize planner task paths before L2 lookup.~~ Implemented with one shared
    workspace-relative normalizer used at planner validation and every L2 path
    boundary.
14. ~~Correct planner-handoff evaluation-attempt counts.~~ Implemented by counting
    every completed PASS or FAIL invocation while retaining `reflection_count` as
    the failure counter.
15. ~~Remove current-task conditioning from L2 summaries.~~ Implemented; L2 prompts
    now contain only the file path, full current content, and task-independent
    summary contract.
16. ~~Bound content-bearing tool arguments in formatted L1.~~ Ignored as an
    accepted evidence cost for L3 extraction.
17. ~~Add L3 reconciliation and a deterministic relevance policy.~~ Implemented with
    full-list mutation and an all-results-above-`0.60` retrieval threshold.
18. ~~Isolate compaction from L2/L3 injection.~~ Implemented with raw task-local L1
    as the only material being compacted.
19. ~~Include successful evaluator evidence in separate-memory L3.~~ Implemented
    with labeled programmer and evaluator task-local logs.
20. ~~Include relevant files in L3 retrieval queries.~~ Implemented.
21. ~~Stop retrying deterministic unsupported L2 paths.~~ Implemented while
    retaining transient read failures for later retries.
22. ~~Retry malformed L2 summaries immediately.~~ Implemented with one
    schema-corrected retry that remains subject to the internal monetary limit.
23. ~~Prevent think-tag cleanup from corrupting content and L1.~~ Implemented by
    splitting reasoning at the API boundary and consuming final content directly;
    missing-final-brace repair remains as independent malformed-JSON tolerance.

All identified issues are now resolved or explicitly accepted/ignored. The accepted
design costs above should remain disclosed in the experimental report.
