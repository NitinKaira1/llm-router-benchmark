

import os


MODEL_DIR = os.environ.get("MODEL_DIR", r"C:\Users\pushp\Desktop\model")

LLAMA_SERVER_EXE = os.environ.get(
    "LLAMA_SERVER_EXE",
    r"C:\Users\pushp\Desktop\model\llama-b9606-bin-win-cpu-x64\llama-server.exe",
)


N_CTX = int(os.environ.get("LLAMA_N_CTX", 4096))          
N_THREADS = int(os.environ.get("LLAMA_N_THREADS", os.cpu_count() or 4))
SERVER_STARTUP_TIMEOUT_S = 120  # how long to wait for a model to finish loading


MODELS = {
    "tiny": {
        "file": "qwen3-1.7b-instruct-q4_k_m.gguf",
        "label": "Qwen3 1.7B Instruct",
        "port": 8081,
        "price_in": 0.02,
        "price_out": 0.02,
    },
    "small": {
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "label": "Qwen2.5 3B Instruct",
        "port": 8082,
        "price_in": 0.04,
        "price_out": 0.04,
    },
    "mid": {
        "file": "qwen3-4b-instruct-2507-q4_k_m.gguf",
        "label": "Qwen3 4B Instruct (2507)",
        "port": 8083,
        "price_in": 0.06,
        "price_out": 0.06,
    },
    "large": {
        "file": "qwen3-8b-q4_k_m.gguf",
        "label": "Qwen3 8B",
        "port": 8084,
        "price_in": 0.12,
        "price_out": 0.12,
    },
}

CLASSIFIER_MODEL = "tiny"

JUDGE_MODEL = "large"


BASELINE_MODEL = "large"


ROUTING_MAP = {
    "simple": "tiny",
    "code": "small",
    "creative": "mid",
    "reasoning": "large",
}
