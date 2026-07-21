
import json
from config import JUDGE_MODEL
from models import call_model
from text_utils import strip_think_tags

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator grading the quality of an AI assistant's response.

Score the response on a scale of 1-5 based on:
- Correctness / factual accuracy
- Completeness (does it fully answer what was asked)
- Clarity and coherence

Scoring guide:
5 = Excellent: fully correct, complete, clear
4 = Good: correct with minor gaps or slight awkwardness
3 = Acceptable: mostly correct but missing something notable
2 = Poor: significant errors or incomplete
1 = Very poor: wrong, irrelevant, or nonsensical

Respond with ONLY a JSON object, no other text:
{"score": <1-5 integer>, "reasoning": "<one short sentence>"}

/no_think"""


def judge(prompt: str, response: str) -> dict:
    judge_prompt = f"Original prompt:\n{prompt}\n\nAssistant's response:\n{response}\n\nScore this response."
    result = call_model(JUDGE_MODEL, judge_prompt, system=JUDGE_SYSTEM_PROMPT, max_tokens=100)
    raw = strip_think_tags(result["text"])

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 3))
        reasoning = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        score = 3
        reasoning = "Judge output could not be parsed; defaulted to 3."

    score = max(1, min(5, score))  # clamp to valid range

    return {
        "score": score,
        "reasoning": reasoning,
        "judge_call": result,  # judge's own cost/latency, tracked separately, not counted against router
    }
