# LLM Router: Cost/Quality Tradeoffs, Benchmarked (100% Local)

A working router that classifies incoming prompts by complexity/type and
sends each one to the smallest local model that can handle it well —
instead of always calling the biggest one. Benchmarked against an
"always use the big model" baseline on 36 diverse prompts, scored by an
LLM judge. **No API keys, no internet required** — everything runs on
your machine via `llama-cpp-python` against your own GGUF files.

## Your models

Point `MODEL_DIR` at the folder containing these 4 files:

| Tier | File | Size |
|---|---|---|
| tiny | `qwen3-1.7b-instruct-q4_k_m.gguf` | ~1.08 GB |
| small | `qwen2.5-3b-instruct-q4_k_m.gguf` | ~2.06 GB |
| mid | `qwen3-4b-instruct-2507-q4_k_m.gguf` | ~2.44 GB |
| large | `qwen3-8b-q4_k_m.gguf` | ~4.91 GB |

## Why this exists

Every team serving LLM traffic at scale runs into the same waste: a
"what's the capital of France" query costs the same compute as a
30-step reasoning problem if you route everything to your best model.
Products like OpenRouter, Martian, and NotDiamond exist to solve exactly
this. This is a small, from-scratch version of that idea, running
entirely on local hardware, with real measured numbers attached.

## How it works

1. **Classify** — the smallest model (Qwen3 1.7B) tags the incoming
   prompt as `simple`, `code`, `reasoning`, or `creative` using a
   few-shot prompt (see `classifier.py`).
2. **Route** — the label maps to a model tier (`config.ROUTING_MAP`):
   simple → tiny (1.7B), code → small (3B), creative → mid (4B),
   reasoning → large (8B).
3. **Answer** — the chosen model generates the actual response.
4. **Judge** — the largest model (Qwen3 8B) scores every response 1-5
   for correctness, completeness, and clarity.
5. **Compare** — every prompt runs through both the router path and a
   baseline path (always Qwen3 8B), logging real measured cost,
   latency, and tokens/sec for both.

## Setup

No compiler needed — this uses the prebuilt `llama-server.exe` from a
llama.cpp release (e.g. `llama-b9606-bin-win-cpu-x64`), not the
`llama-cpp-python` pip package.

```bash
pip install -r requirements.txt
```

Then set two paths — either edit them directly at the top of
`config.py`, or set as environment variables:

```powershell
$env:MODEL_DIR = "C:\Users\pushp\Desktop\model"
$env:LLAMA_SERVER_EXE = "C:\Users\pushp\Desktop\model\llama-b9606-bin-win-cpu-x64\llama-server.exe"
```

If you're not sure of the exact `llama-server.exe` path, find it with:
```powershell
dir "C:\Users\pushp\Desktop\model\llama-b9606-bin-win-cpu-x64" -Recurse -Filter "llama-server.exe"
```

Run the smoke test first to confirm everything's wired up before
committing to the full 36-prompt benchmark:

```bash
python smoke_test.py
```

## Running it

```bash
python benchmark.py        # starts all 4 llama-server instances, runs 36 prompts through both paths
streamlit run dashboard.py # opens the results dashboard
```

Each tier gets its own `llama-server.exe` subprocess on its own port
(8081-8084), started automatically the first time that tier is called
and reused after that (`models.py` handles this — `preload_all()`
starts all 4 up front so mid-benchmark tier switches don't pay a
cold-start cost). Servers are terminated automatically when the script
exits.

Expect the full 36-prompt run to take anywhere from a few minutes to
~20+ minutes on CPU — the 8B model is the slow one. If your machine
can't comfortably hold all 4 models in RAM at once (~10.5GB total for
these quantized files), remove the `preload_all()` call in
`benchmark.py` so servers start lazily on demand instead of all at once.

## About the "cost saved" numbers

These models run locally for **$0** — there's no metered API. The
`price_in`/`price_out` figures in `config.py` are reference prices for
similarly-sized open-weight models on a paid hosting provider, included
only so the benchmark can still report a meaningful "$ saved" story, as
if these were served commercially instead of run on your machine. If
you'd rather report pure compute-time savings instead of a dollar
figure, the dashboard's latency and tokens/sec charts already do that
with real measured numbers — no simulation needed there.

## Results

*(Fill this in after running `benchmark.py` — the numbers below are
placeholders showing the shape of the output.)*

> Routing cut equivalent API cost by **XX%** and wall-clock time by
> **XX%**, with a quality change of **+/-X.X points** (1-5 scale)
> across 36 prompts, running entirely on local hardware.

| Metric | Baseline (always Qwen3 8B) | Router |
|---|---|---|
| Total cost (reference) | $X.XXXX | $X.XXXX |
| Total latency | XX.Xs | XX.Xs |
| Avg quality score | X.XX / 5 | X.XX / 5 |
| Classifier accuracy | — | XX% |

![dashboard screenshot](docs/dashboard_screenshot.png)

## Project structure

```
llm_router/
├── config.py       # model tiers, MODEL_DIR, LLAMA_SERVER_EXE, routing map
├── models.py        # launches llama-server.exe per tier, calls it over HTTP, tracks real latency/tok-per-sec
├── smoke_test.py      # quick check that paths + one model call work before the full run
├── classifier.py       # few-shot complexity/type classifier
├── router.py             # ties classification + model call together
├── judge.py                # LLM-as-judge scoring (1-5)
├── dataset.py                # 36 test prompts across 4 categories
├── benchmark.py                # runs baseline vs router, saves results/benchmark_results.json
├── dashboard.py                 # Streamlit dashboard (Plotly charts)
└── results/
    └── benchmark_results.json
```

## Performance tips (CPU is slow — this matters)

- Run `python benchmark.py --quick` first — it only runs 6 prompts, so
  you find out fast if something's misconfigured instead of waiting
  20+ minutes into a full run.
- The classifier and judge have Qwen3's "thinking mode" disabled
  (`/no_think` in their system prompts) and a small `max_tokens` cap,
  since they only need to output one line of JSON — thinking mode was
  previously burning hundreds of extra tokens on every single call,
  which is the #1 cause of a benchmark run feeling stuck.
- The actual answer-generation calls (baseline + router) still allow
  Qwen3 to think by default, since that's arguably part of why the
  "reasoning" tier deserves to be the expensive one. If you want
  faster runs at the cost of some reasoning quality, add `/no_think`
  to the end of the prompt in `router.py` and `benchmark.py`'s baseline
  call too.
- `benchmark.py` now prints per-step timing (baseline answer, baseline
  judge, router answer, router judge) so you can see exactly where
  time is going per prompt instead of it looking frozen.

## Honest limitations

- 36 prompts is a demo-scale sample, not a statistically powered
  benchmark — treat the percentages as directional, not definitive.
- The judge model (Qwen3 8B) is also the "large" tier being graded in
  the baseline, which risks some self-preference bias — same model
  grading its own baseline answers. Worth calling out explicitly in
  any writeup; a more rigorous version would use a separate, larger,
  non-competing judge model.
- Classifier errors compound: a `reasoning` query misclassified as
  `simple` gets under-routed and likely scores worse — that's real
  router behavior, not a bug, but worth calling out.
- "Cost saved" is reference/list pricing for comparable hosted models,
  not your actual spend, which is $0 either way since this runs locally.
- Quantized (`q4_k_m`) models trade some quality for speed/size vs. the
  full-precision originals — that's part of the story too, not hidden.
