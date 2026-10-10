# My-Mem: minimal influence-validation prototype

My-Mem is a small research fork built around one question:

> When candidate memories are already available, does explicitly constructing
> and validating their possible influence improve the final decision over
> passing the memories directly to an LLM?

The repository deliberately separates two layers:

- `T_mem/`: selected T-Mem memory construction and retrieval code.  It is the
  candidate generator and should remain close to upstream for fair comparison.
- `my_mem/`: the new
  `TaskFrame -> Candidate -> Hypothesis -> Gap -> Evidence -> Validation`
  pipeline.

The first prototype intentionally excludes factor ontologies, memory write-back,
learned execution policies, user questioning and open-ended multi-round search.
Those extensions have interfaces or TODO notes but are not prerequisites for the
first experiment.

## Minimal flow

1. Run the retained T-Mem build/retrieval stages with `scripts/build_memory.sh`.
   The forked Stage 6 additionally writes a structured `candidates` array beside
   the original prompt context.
2. Convert one Stage 6 record into a `TaskCase` with
   `my_mem.adapters.tmem.load_cases`.
3. Run `python -m my_mem.main.run_minimal --input CASES.json --output RUN.json`.
4. Compare the resulting validated influences with a direct-answer baseline
   under the same model and token/call budget.

直接读取 T-Mem Stage 6 输出时，可以运行：

```bash
python -m my_mem.main.run_minimal \
  --input results/<run>/search_results.json \
  --input-format tmem \
  --evidence-pool evidence_pool.json \
  --output results/influence_results.json
```

生成模型默认读取 `T_MEM_LLM_BASE_MODEL`。`--model` 仅用于临时覆盖，正常
实验不需要传入，以免命令行模型名与环境配置不一致。

The original LoCoMo/LoCoMo-Plus data are not duplicated into this directory.
Pass their paths with `--locomo-file` / `--locomo-plus-file`, or point to data
stored under `T-Mem_origin/benchmark_eval/...`.

## Environment

The reused T-Mem provider accepts OpenAI-compatible endpoints. See
`.env.example`. Chat, embedding and reranking model choices are centralized in
environment variables, so sourcing `cloud.env` or `local.env` switches the
whole pipeline without changing code. Embedding reuses the LLM endpoint and API
key unless explicitly overridden; reranking may use a provider-specific full
URL through `T_MEM_RERANKER_URL`.

## Current boundaries

- Retrieval metadata helps locate candidates but is never evidence.
- Influence is classified natively as `high`, `low`, or `none`; `impact` is a
  within-level ranking score with schema-enforced ranges (`none < 0.20`,
  `0.20 <= low < 0.60`, `high >= 0.60`).
- Facts already stated by the task or candidate memory are recorded as
  supported claims and must not be proposed again as evidence gaps.
- A query-only TaskFrame supplies common dimensions and prioritized response
  obligations. Each candidate may add at most one temporary, grounded dimension;
  it is not written back to the common frame and does not trigger another recall.
- Gap and document sources name acquisition channels only:
  `personal_memory_store`, `external_knowledge`, `environment_tool`, or `user`.
- The whole influence bridge is expanded in one hypothesis call. Potentially
  high-impact, low-confidence bridges are logged for later analysis rather than
  triggering another model or candidate-recall round.
- Every acquired evidence record carries provenance.
- A gap is a testable proposition, not a free-form statement of uncertainty.
- The deterministic controller enforces candidate, gap and evidence budgets.
- Inference results are not written back to long-term memory in this version.
