

from classifier import classify
from models import call_model


def route(prompt: str) -> dict:
    """
    Full router path for a single prompt. Returns a dict with:
      - text: final answer
      - label: classifier's predicted query type
      - tier: model tier used to answer
      - total_cost_usd: classifier cost + answer cost
      - total_latency_s: classifier latency + answer latency
      - answer_call: raw call dict for the answering model
      - classify_result: raw classifier output
    """
    classify_result = classify(prompt)
    tier = classify_result["tier"]

    answer_call = call_model(tier, prompt)

    total_cost = classify_result["classifier_call"]["cost_usd"] + answer_call["cost_usd"]
    total_latency = classify_result["classifier_call"]["latency_s"] + answer_call["latency_s"]

    return {
        "text": answer_call["text"],
        "label": classify_result["label"],
        "confidence": classify_result["confidence"],
        "tier": tier,
        "model_id": answer_call["model_id"],
        "total_cost_usd": total_cost,
        "total_latency_s": total_latency,
        "answer_call": answer_call,
        "classify_result": classify_result,
    }
