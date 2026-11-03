# Evaluate an LLM with Opik

One script that evaluates any model behind an **OpenAI-compatible endpoint** on two public benchmarks, scores every answer with heuristic and LLM-judge metrics, and logs everything to [Opik](https://www.comet.com/opik) so you can inspect each answer and compare models side by side.

| Benchmark | What it tests | Dataset |
|---|---|---|
| `gsm8k` | Multi-step reasoning: grade-school math word problems with a single numeric answer | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (first 200 test problems) |
| `humaneval` | Code generation: complete a Python function, scored by running its unit tests | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (164 problems) |

## Metrics

| Metric | Type | Benchmark | What it measures |
|---|---|---|---|
| `correct_answer` | heuristic | gsm8k | Last number in the reply exactly equals the reference answer |
| `follows_answer_format` | heuristic | gsm8k | Reply ends with `Answer: <number>` as instructed |
| `passes_unit_tests` | heuristic | humaneval | Generated code passes HumanEval's unit tests (pass@1) |
| `valid_python` | heuristic | humaneval | Extracted code parses as Python |
| `uses_code_block` | heuristic | humaneval | Reply puts the code in a fenced Python block as instructed |
| `judge_quality` | LLM judge (G-Eval) | both | Math: is the reasoning sound compared with the reference solution? Code: correct, idiomatic, handles edge cases? |
| `answer_relevance` | LLM judge | both | Does the reply address the question? |
| `usefulness` | LLM judge | both | How helpful is the reply? |

Every score comes with a reason (e.g. `answered 18, expected 18`, `AssertionError`, or the judge's explanation), visible per row in the Opik UI.

## Setup

**1. Opik account.** Create a free account at [comet.com/opik](https://www.comet.com/opik). Copy your API key from the profile menu and note your workspace name.

**2. Install and connect.**

```bash
pip install -r requirements.txt
opik configure        # asks for your Opik API key and workspace, saves ~/.opik.config
```

In CI or another non-interactive shell, skip `opik configure` and set `OPIK_API_KEY` and `OPIK_WORKSPACE` instead.

**3. Install the Opik MCP server (recommended).** This lets your AI coding assistant (Claude Code, Cursor, VS Code, Codex, ...) read your Opik experiments and traces directly: ask it things like *"compare my last two humaneval experiments and show me the problems that regressed"* or *"why did this trace fail?"*.

```bash
opik mcp configure                 # detects your AI clients and registers the server
opik mcp status                    # check which clients are configured
```

Without a terminal, name the client: `opik mcp configure --ai-client cursor`. You can also do both steps at once with `opik configure --install-mcp`.

**4. Point at your model.**

```bash
export MODEL_API_KEY=<key for your endpoint>
python evaluate.py --base-url https://<your-endpoint>/v1 --list-models   # see available model IDs
```

## Run

```bash
# Quick first pass: 10 items per benchmark
python evaluate.py --base-url https://<your-endpoint>/v1 --model <model-id> --limit 10

# Full run, one benchmark
python evaluate.py --base-url https://<your-endpoint>/v1 --model <model-id> --benchmark humaneval
```

The script prints a summary table per benchmark and a link to each experiment.

### Using a different judge model

By default the LLM-judge metrics use the model under test, on the same endpoint. To grade with a different (usually stronger) model, possibly on a different endpoint:

```bash
export JUDGE_API_KEY=<key for the judge endpoint>
python evaluate.py \
  --base-url https://<your-endpoint>/v1 --model <model-id> \
  --judge-base-url https://<judge-endpoint>/v1 --judge-model <judge-model-id> \
  --limit 10
```

Everything can also be set through environment variables: `MODEL_BASE_URL`, `MODEL`, `MODEL_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY`.

### Options

| Flag | Default | |
|---|---|---|
| `--base-url` | `$MODEL_BASE_URL` | OpenAI-compatible base URL of the model under test (required) |
| `--model` | `$MODEL` | Model ID (required) |
| `--api-key-env` | `MODEL_API_KEY` | Name of the env var holding the endpoint's API key |
| `--judge-base-url` / `--judge-model` | same as model | Endpoint and model for the LLM judges |
| `--judge-api-key-env` | `JUDGE_API_KEY` | Falls back to the model's key |
| `--benchmark` | `all` | `gsm8k`, `humaneval`, or `all` |
| `--limit` | all items | Evaluate only the first N items per benchmark |
| `--project` | `llm-eval-quickstart` | Opik project for datasets, prompts and traces |
| `--no-judges` | | Heuristic metrics only (no judge calls) |
| `--no-exec` | | Don't execute generated code (skips `passes_unit_tests`) |
| `--threads` | `8` | Parallel requests to the endpoint |
| `--list-models` | | Print the endpoint's model IDs and exit |

> **Note:** `passes_unit_tests` executes model-generated code on your machine, in a subprocess with a 15-second timeout. Run inside a container or VM if that's a concern, or pass `--no-exec`.

## Read the results

Each run creates one experiment per benchmark, named `<model>-<benchmark>` and tagged with both, so runs are easy to find. In the Opik UI you get:

- the aggregate score for each metric,
- every row's input, model output, extracted answer or code, scores with reasons, and the full trace (including the judge calls),
- the prompt used, versioned in Opik's prompt library and linked to the experiment,
- the model and judge configuration for the run.

To compare models, rerun with a different `--model` and open both experiments together for a side-by-side view and per-row diffs. If you set up the MCP server, you can ask your AI assistant to do this comparison for you.

## Bring your own data

Each benchmark is an entry in the `BENCHMARKS` dict in `evaluate.py`: a dataset loader, a prompt template, a task function and its metrics. To evaluate on an internal golden set, copy the `gsm8k` entry, point `load_items` at your data (each item needs an `input`, plus whatever your metrics compare against), and adjust the prompt and metrics.

Full docs: [Evaluate your LLM](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm).
