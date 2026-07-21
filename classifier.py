

import json
from config import CLASSIFIER_MODEL, ROUTING_MAP
from models import call_model
from text_utils import strip_think_tags

CLASSIFIER_SYSTEM_PROMPT = """You are a query complexity classifier for an LLM router.
Classify the user's query into EXACTLY ONE of these four labels:

- simple: factual recall, definitions, basic math/conversions, short lookups,
  small talk, single-step questions with an unambiguous answer.
- code: writing, debugging, explaining, or reviewing code in any language.
- reasoning: multi-step logic, word problems, planning, tricky analysis,
  math proofs, puzzles, anything requiring careful chained reasoning.
- creative: open-ended writing, brainstorming, stories, marketing copy,
  poems, ideation.

Respond with ONLY a JSON object, no other text, in this exact format:
{"label": "<one of: simple, code, reasoning, creative>", "confidence": <0-1 float>}

Examples:
Query: "What is the capital of Japan?"
{"label": "simple", "confidence": 0.98}

Query: "Write a Python function to reverse a linked list in place."
{"label": "code", "confidence": 0.95}

Query: "A train leaves Chicago at 60mph, another leaves NY at 80mph 900 miles apart, when do they meet?"
{"label": "reasoning", "confidence": 0.9}

Query: "Write a short poem about autumn rain."
{"label": "creative", "confidence": 0.95}

Query: "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain the logic."
{"label": "reasoning", "confidence": 0.92}

/no_think"""


def classify(prompt: str) -> dict:
    """Returns {"label": str, "confidence": float, "tier": str, "classifier_call": dict}"""
    result = call_model(CLASSIFIER_MODEL, prompt, system=CLASSIFIER_SYSTEM_PROMPT, max_tokens=60)
    raw = strip_think_tags(result["text"])

   
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        label = parsed.get("label", "simple").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError):

        label = "simple"
        confidence = 0.0

    if label not in ROUTING_MAP:
        label = "simple"

    tier = ROUTING_MAP[label]

    return {
        "label": label,
        "confidence": confidence,
        "tier": tier,
        "classifier_call": result,  # keep for cost/latency accounting
    }
