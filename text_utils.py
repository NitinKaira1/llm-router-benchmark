
import re


def strip_think_tags(text: str) -> str:

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()
