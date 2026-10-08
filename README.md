# My-Mem: minimal influence-validation prototype

My-Mem is a small research fork built around one question:

> When candidate memories are already available, does explicitly constructing
> and validating their possible influence improve the final decision over
> passing the memories directly to an LLM?

The repository deliberately separates two layers:

- `T_mem/`: selected T-Mem memory construction and retrieval code.  It is the
  candidate generator and should remain close to upstream for fair comparison.
- `my_mem/`: the new `Candidate -> Hypothesis -> Gap -> Evidence -> Validation`
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

`examples/sample_cases.json` is a local, deterministic smoke-test input.  The
fixed evidence pool is only a controlled experimental substitute for future
memory, knowledge, environment and user evidence providers.

The original LoCoMo/LoCoMo-Plus data are not duplicated into this directory.
Pass their paths with `--locomo-file` / `--locomo-plus-file`, or point to data
stored under `T-Mem_origin/benchmark_eval/...`.

## Environment

The reused T-Mem provider accepts OpenAI-compatible chat-completion endpoints.
See `.env.example`.  My-Mem uses the existing `T_MEM_*` variables for now so the
candidate generator and influence layer share one reproducible model setup.

## Current boundaries

- Retrieval metadata helps locate candidates but is never evidence.
- Every acquired evidence record carries provenance.
- A gap is a testable proposition, not a free-form statement of uncertainty.
- The deterministic controller enforces candidate, gap and evidence budgets.
- Inference results are not written back to long-term memory in this version.

