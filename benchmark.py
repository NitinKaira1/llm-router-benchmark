
import json
import time
import sys
from datetime import datetime

from dataset import TEST_PROMPTS
from config import BASELINE_MODEL, MODELS
from models import call_model, preload_all
from router import route
from judge import judge
from text_utils import strip_think_tags

RESULTS_PATH = "results/benchmark_results.json"


def run_benchmark(quick: bool = False):
    prompts = TEST_PROMPTS[::6] if quick else TEST_PROMPTS
    if quick:
        print("Running in --quick mode: 6 prompts only.\n")

    print(f"Loading all {len(MODELS)} local models up front (one-time cost)...")
    t0 = time.time()
    preload_all()
    print(f"All models loaded in {time.time()-t0:.1f}s. Starting benchmark.\n")

    results = []
    n = len(prompts)

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        expected_label = item["expected_label"]
        prompt_start = time.time()
        print(f"[{i+1}/{n}] {prompt[:60]}...")


        step_t = time.time()
        try:
            baseline_call = call_model(BASELINE_MODEL, prompt)
        except Exception as e:
            print(f"  ! baseline call failed: {e}")
            continue
        print(f"    baseline answer: {baseline_call['latency_s']:.1f}s ({baseline_call['output_tokens']} tokens)")

        baseline_clean_text = strip_think_tags(baseline_call["text"])

        step_t = time.time()
        try:
            baseline_judged = judge(prompt, baseline_clean_text)
        except Exception as e:
            print(f"  ! baseline judge failed: {e}")
            baseline_judged = {"score": None, "reasoning": f"judge error: {e}"}
        print(f"    baseline judge:  {time.time()-step_t:.1f}s")

        step_t = time.time()
        try:
            router_result = route(prompt)
        except Exception as e:
            print(f"  ! router call failed: {e}")
            continue
        print(f"    router → {router_result['tier']:9s} answer: {router_result['total_latency_s']:.1f}s")

        router_clean_text = strip_think_tags(router_result["text"])

        step_t = time.time()
        try:
            router_judged = judge(prompt, router_clean_text)
        except Exception as e:
            print(f"  ! router judge failed: {e}")
            router_judged = {"score": None, "reasoning": f"judge error: {e}"}
        print(f"    router judge:    {time.time()-step_t:.1f}s")
        print(f"    total for this prompt: {time.time()-prompt_start:.1f}s\n")

        classifier_correct = router_result["label"] == expected_label

        results.append({
            "prompt": prompt,
            "expected_label": expected_label,
            "baseline": {
                "model_tier": BASELINE_MODEL,
                "model_id": baseline_call["model_id"],
                "cost_usd": baseline_call["cost_usd"],
                "latency_s": baseline_call["latency_s"],
                "tokens_per_sec": baseline_call["tokens_per_sec"],
                "quality_score": baseline_judged["score"],
                "quality_reasoning": baseline_judged["reasoning"],
                "text": baseline_clean_text,
                "text_raw": baseline_call["text"],
            },
            "router": {
                "predicted_label": router_result["label"],
                "classifier_correct": classifier_correct,
                "model_tier": router_result["tier"],
                "model_id": router_result["model_id"],
                "cost_usd": router_result["total_cost_usd"],
                "latency_s": router_result["total_latency_s"],
                "tokens_per_sec": router_result["answer_call"]["tokens_per_sec"],
                "quality_score": router_judged["score"],
                "quality_reasoning": router_judged["reasoning"],
                "text": router_clean_text,
                "text_raw": router_result["text"],
            },
        })




    n_done = len(results)
    total_baseline_cost = sum(r["baseline"]["cost_usd"] for r in results)
    total_router_cost = sum(r["router"]["cost_usd"] for r in results)
    total_baseline_latency = sum(r["baseline"]["latency_s"] for r in results)
    total_router_latency = sum(r["router"]["latency_s"] for r in results)

    baseline_scores = [r["baseline"]["quality_score"] for r in results if r["baseline"]["quality_score"] is not None]
    router_scores = [r["router"]["quality_score"] for r in results if r["router"]["quality_score"] is not None]

    avg_baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
    avg_router_score = sum(router_scores) / len(router_scores) if router_scores else None

    classifier_accuracy = sum(1 for r in results if r["router"]["classifier_correct"]) / n_done if n_done else None

    summary = {
        "n_prompts": n_done,
        "total_baseline_cost_usd": total_baseline_cost,
        "total_router_cost_usd": total_router_cost,
        "cost_saved_pct": (1 - total_router_cost / total_baseline_cost) * 100 if total_baseline_cost else None,
        "total_baseline_latency_s": total_baseline_latency,
        "total_router_latency_s": total_router_latency,
        "latency_saved_pct": (1 - total_router_latency / total_baseline_latency) * 100 if total_baseline_latency else None,
        "avg_baseline_quality": avg_baseline_score,
        "avg_router_quality": avg_router_score,
        "quality_delta": (avg_router_score - avg_baseline_score) if (avg_baseline_score and avg_router_score) else None,
        "classifier_accuracy": classifier_accuracy,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    output = {"summary": summary, "results": results}

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    quick_mode = "--quick" in sys.argv
    run_benchmark(quick=quick_mode)
