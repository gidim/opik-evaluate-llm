"""
End-to-end Opik evaluation of any OpenAI-compatible model endpoint on two
public benchmarks:

  gsm8k      grade-school math word problems (reasoning, single numeric answer)
  humaneval  Python function completion, scored by running unit tests

    pip install -r requirements.txt
    opik configure                       # once: Opik API key + workspace
    export MODEL_API_KEY=...             # key for the endpoint under test
    python evaluate.py --base-url https://<your-endpoint>/v1 --model <model-id> --limit 10

Run with --help for all options, and --list-models to see which model IDs the
endpoint serves.

WARNING: the humaneval `passes_unit_tests` metric executes model-generated code
on this machine (in a subprocess with a timeout). Run inside a container or VM
if that is a concern, or pass --no-exec to skip it.
"""

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List

import openai
import opik
from datasets import load_dataset
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    GEval,
    RegexMatch,
    Usefulness,
    score_result,
)
from opik.evaluation.models import LiteLLMChatModel
from opik.integrations.openai import track_openai

EXEC_TIMEOUT_SECONDS = 15


# =========================================================================== config

def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("model under test")
    g.add_argument("--model", default=env("MODEL"), help="model ID (env: MODEL)")
    g.add_argument("--base-url", default=env("MODEL_BASE_URL"),
                   help="OpenAI-compatible base URL, e.g. https://<your-endpoint>/v1 (env: MODEL_BASE_URL, required)")
    g.add_argument("--api-key-env", default="MODEL_API_KEY",
                   help="env var holding the API key (default: MODEL_API_KEY)")

    g = p.add_argument_group("judge model (defaults to the model under test)")
    g.add_argument("--judge-model", default=env("JUDGE_MODEL"), help="judge model ID (env: JUDGE_MODEL)")
    g.add_argument("--judge-base-url", default=env("JUDGE_BASE_URL"), help="judge base URL (env: JUDGE_BASE_URL)")
    g.add_argument("--judge-api-key-env", default="JUDGE_API_KEY",
                   help="env var holding the judge API key (default: JUDGE_API_KEY, falls back to the model key)")

    g = p.add_argument_group("run")
    g.add_argument("--benchmark", choices=["all", *BENCHMARKS], default="all", help="which benchmark(s) to run (default: all)")
    g.add_argument("--limit", type=int, default=None, help="evaluate only the first N items of each benchmark")
    g.add_argument("--project", default="llm-eval-quickstart", help="Opik project for datasets, prompts and traces")
    g.add_argument("--no-judges", action="store_true", help="skip LLM-judge metrics (heuristic metrics only)")
    g.add_argument("--no-exec", action="store_true", help="don't execute generated code (skips passes_unit_tests)")
    g.add_argument("--threads", type=int, default=8, help="parallel requests to the endpoint (default: 8)")
    g.add_argument("--list-models", action="store_true", help="print the endpoint's model IDs and exit")

    args = p.parse_args()
    args.api_key = env(args.api_key_env)
    if not args.api_key:
        p.error(f"set {args.api_key_env} to your endpoint's API key")
    if not args.base_url:
        p.error("--base-url is required (or set MODEL_BASE_URL), e.g. https://<your-endpoint>/v1")
    if not args.list_models and not args.model:
        p.error("--model is required (run with --list-models to see what the endpoint serves)")
    args.judge_model = args.judge_model or args.model
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_api_key = env(args.judge_api_key_env) or args.api_key
    return args


# =========================================================================== shared helpers

def numeric_metric(name, ok, reason):
    return score_result.ScoreResult(name=name, value=float(ok), reason=reason)


def make_judge_metrics(judge, review_intro, review_criteria, render_for_review):
    """LLM-judge metrics shared by both benchmarks: a task-specific G-Eval rubric plus two generic judges."""
    geval = GEval(model=judge, name="judge_quality",
                  task_introduction=review_intro, evaluation_criteria=review_criteria)

    def judge_quality(dataset_item, task_outputs):
        return geval.score(output=render_for_review(dataset_item, task_outputs))

    return [judge_quality], [
        AnswerRelevance(model=judge, require_context=False, name="answer_relevance"),
        Usefulness(model=judge, name="usefulness"),
    ]


# =========================================================================== gsm8k

GSM8K_PROMPT = """Solve the following math problem. Think step by step, then end your reply with a final line of the form "Answer: <number>".

{{question}}"""


def gsm8k_items():
    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(200))
    return [
        {
            "input": r["question"],
            "expected_output": r["answer"].split("####")[-1].strip().replace(",", ""),
            "reference_solution": r["answer"].split("####")[0].strip(),
        }
        for r in rows
    ]


def gsm8k_task(prompt):
    def task(item, llm, model):
        reply = chat(llm, model, prompt.format(question=item["input"]))
        numbers = re.findall(r"-?\d+(?:\.\d+)?", reply.replace(",", ""))
        return {"input": item["input"], "output": reply, "answer": numbers[-1] if numbers else None}
    return task


def correct_answer(dataset_item, task_outputs):
    """Exact numeric match between the last number in the reply and the reference answer."""
    got, want = task_outputs["answer"], dataset_item["expected_output"]
    ok = got is not None and float(got) == float(want)
    return numeric_metric("correct_answer", ok, f"answered {got}, expected {want}")


def gsm8k_metrics(judge):
    scoring_functions = [correct_answer]
    scoring_metrics = [RegexMatch(regex=r"Answer:\s*\$?-?[\d,]+(\.\d+)?\s*$", name="follows_answer_format")]
    if judge:
        fns, metrics = make_judge_metrics(
            judge,
            review_intro="You are a math teacher grading a student's worked solution to a word problem.",
            review_criteria=("Compare the STUDENT SOLUTION to the REFERENCE SOLUTION. Score highly if the reasoning "
                             "is correct, every step follows from the previous one, and the final answer matches. "
                             "Score low for arithmetic errors, unjustified leaps, or a wrong final answer."),
            render_for_review=lambda item, out: (f"PROBLEM:\n{item['input']}\n\nREFERENCE SOLUTION:\n"
                                                 f"{item['reference_solution']}\n\nSTUDENT SOLUTION:\n{out['output']}"),
        )
        scoring_functions += fns
        scoring_metrics += metrics
    return scoring_functions, scoring_metrics


# =========================================================================== humaneval

HUMANEVAL_PROMPT = """Complete the following Python function.
Reply with the full function (including any imports it needs) in a single ```python code block.

{{code_stub}}"""


def humaneval_items():
    rows = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "input": r["prompt"],
            "task_id": r["task_id"],
            "entry_point": r["entry_point"],
            "test": r["test"],
            "expected_output": r["canonical_solution"],
        }
        for r in rows
    ]


def extract_code(text):
    """Return the last ```python block in the reply, or the whole reply if there is none."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text


def humaneval_task(prompt):
    def task(item, llm, model):
        reply = chat(llm, model, prompt.format(code_stub=item["input"]))
        return {"input": item["input"], "output": reply, "code": extract_code(reply)}
    return task


def passes_unit_tests(dataset_item, task_outputs):
    """Functional correctness (pass@1): run HumanEval's unit tests against the generated code."""
    # The original stub goes first so its imports/helpers exist; the model's definition overrides it.
    program = "\n".join([dataset_item["input"], task_outputs["code"], dataset_item["test"],
                         f"check({dataset_item['entry_point']})"])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run([sys.executable, "-I", path], cwd=tmp, capture_output=True,
                                  text=True, timeout=EXEC_TIMEOUT_SECONDS)
            ok = proc.returncode == 0
            reason = "all tests passed" if ok else (proc.stderr.strip().splitlines() or ["failed"])[-1]
        except subprocess.TimeoutExpired:
            ok, reason = False, f"timed out after {EXEC_TIMEOUT_SECONDS}s"
    return numeric_metric("passes_unit_tests", ok, reason[:500])


def valid_python(dataset_item, task_outputs):
    """Does the extracted code parse?"""
    try:
        ast.parse(task_outputs["code"])
        return numeric_metric("valid_python", True, "parses")
    except SyntaxError as e:
        return numeric_metric("valid_python", False, f"SyntaxError: {e.msg} (line {e.lineno})")


def humaneval_metrics(judge, run_code=True):
    scoring_functions = ([passes_unit_tests] if run_code else []) + [valid_python]
    scoring_metrics = [RegexMatch(regex=r"```(?:python|py)\s*\n", name="uses_code_block")]
    if judge:
        fns, metrics = make_judge_metrics(
            judge,
            review_intro="You are a senior Python engineer reviewing a solution to a programming problem.",
            review_criteria=("Judge the SOLUTION against the PROBLEM. Score highly if it is correct for the stated "
                             "behaviour, handles edge cases, is readable and idiomatic, and has no unnecessary "
                             "complexity. Score low if it is wrong, incomplete, or needlessly convoluted."),
            render_for_review=lambda item, out: f"PROBLEM:\n{item['input']}\n\nSOLUTION:\n{out['code']}",
        )
        scoring_functions += fns
        scoring_metrics += metrics
    return scoring_functions, scoring_metrics


# =========================================================================== benchmark registry

@dataclass
class Benchmark:
    dataset_name: str
    description: str
    prompt_template: str
    load_items: Callable[[], List[dict]]
    make_task: Callable
    make_metrics: Callable


BENCHMARKS = {
    "gsm8k": Benchmark(
        dataset_name="llm-eval-quickstart-gsm8k",
        description="First 200 test problems from GSM8K (openai/gsm8k): grade-school math word problems.",
        prompt_template=GSM8K_PROMPT,
        load_items=gsm8k_items,
        make_task=gsm8k_task,
        make_metrics=lambda judge, args: gsm8k_metrics(judge),
    ),
    "humaneval": Benchmark(
        dataset_name="llm-eval-quickstart-humaneval",
        description="HumanEval (openai/openai_humaneval): 164 Python function-completion problems with unit tests.",
        prompt_template=HUMANEVAL_PROMPT,
        load_items=humaneval_items,
        make_task=humaneval_task,
        make_metrics=lambda judge, args: humaneval_metrics(judge, run_code=not args.no_exec),
    ),
}


# =========================================================================== run

def chat(llm, model, content):
    return llm.chat.completions.create(model=model, messages=[{"role": "user", "content": content}]
                                       ).choices[0].message.content or ""


def run_benchmark(key, args, client, llm, judge):
    bench = BENCHMARKS[key]

    # Dataset: Opik deduplicates items, so re-running the insert is safe.
    dataset = client.get_or_create_dataset(bench.dataset_name, description=bench.description,
                                           project_name=args.project)
    dataset.insert(bench.load_items())
    items = dataset.get_items()
    item_ids = [i["id"] for i in items[:args.limit]] if args.limit else None

    # Prompt: versioned in Opik's prompt library and linked to the experiment.
    prompt = client.create_prompt(name=f"{key}-prompt", prompt=bench.prompt_template, project_name=args.project)

    scoring_functions, scoring_metrics = bench.make_metrics(judge, args)
    task = bench.make_task(prompt)
    model_short = args.model.split("/")[-1]

    print(f"\n=== {key}: {len(item_ids) if item_ids else len(items)} items, model {args.model}")
    result = evaluate(
        experiment_name=f"{model_short}-{key}",
        experiment_config={
            "model": args.model,
            "base_url": args.base_url,
            "judge_model": None if judge is None else args.judge_model,
            "judge_base_url": None if judge is None else args.judge_base_url,
            "benchmark": key,
        },
        experiment_tags=[key, model_short],
        dataset=dataset,
        dataset_item_ids=item_ids,
        task=lambda item: task(item, llm, args.model),
        prompts=[prompt],
        scoring_functions=scoring_functions,
        scoring_metrics=scoring_metrics,
        task_threads=args.threads,
    )
    return key, result.experiment_url


def main():
    args = parse_args()
    endpoint = openai.OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.list_models:
        for m in endpoint.models.list():
            print(m.id)
        return

    judge = None
    if not args.no_judges:
        # "openai/" prefix tells litellm to treat the judge as any OpenAI-compatible API.
        judge = LiteLLMChatModel(model_name=f"openai/{args.judge_model}",
                                 api_base=args.judge_base_url, api_key=args.judge_api_key)

    print(f"Model: {args.model} @ {args.base_url}")
    print(f"Judge: {'(disabled)' if judge is None else f'{args.judge_model} @ {args.judge_base_url}'}")

    client = opik.Opik()
    llm = track_openai(endpoint, project_name=args.project)
    keys = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    links = [run_benchmark(k, args, client, llm, judge) for k in keys]

    print("\nResults:")
    for key, url in links:
        print(f"  {key:10} {url}")


if __name__ == "__main__":
    main()
