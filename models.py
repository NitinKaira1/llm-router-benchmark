
import os
import time
import atexit
import subprocess
import requests

from config import MODELS, MODEL_DIR, LLAMA_SERVER_EXE, N_CTX, N_THREADS, SERVER_STARTUP_TIMEOUT_S

_RUNNING_SERVERS = {}  


def _cost(tier: str, input_tokens: int, output_tokens: int) -> float:
    m = MODELS[tier]
    return (input_tokens / 1_000_000) * m["price_in"] + (output_tokens / 1_000_000) * m["price_out"]


def _cleanup_servers():
    for tier, proc in _RUNNING_SERVERS.items():
        if proc.poll() is None:  # still running
            print(f"Stopping {tier} server (pid {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(_cleanup_servers)


def _ensure_server_running(tier: str):
    """Start llama-server.exe for this tier if it isn't already running,
    and block until it reports healthy."""
    if tier in _RUNNING_SERVERS and _RUNNING_SERVERS[tier].poll() is None:
        return  # already running

    model_path = os.path.join(MODEL_DIR, MODELS[tier]["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Check MODEL_DIR in config.py (or the MODEL_DIR env var)."
        )
    if not os.path.exists(LLAMA_SERVER_EXE):
        raise FileNotFoundError(
            f"llama-server.exe not found: {LLAMA_SERVER_EXE}\n"
            f"Find it with: dir <your llama-b9606 folder> -Recurse -Filter llama-server.exe\n"
            f"then set LLAMA_SERVER_EXE in config.py (or the env var) to that path."
        )

    port = MODELS[tier]["port"]
    print(f"Starting {tier} tier server ({MODELS[tier]['file']}) on port {port}...")

    cmd = [
        LLAMA_SERVER_EXE,
        "-m", model_path,
        "--port", str(port),
        "-c", str(N_CTX),
        "-t", str(N_THREADS),
        "--host", "127.0.0.1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _RUNNING_SERVERS[tier] = proc

    health_url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < SERVER_STARTUP_TIMEOUT_S:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server.exe for tier '{tier}' exited early (code {proc.returncode}). "
                f"Try running the command manually to see the error:\n  {' '.join(cmd)}"
            )
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print(f"  {tier} tier ready.")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError(
        f"{tier} tier server did not become healthy within {SERVER_STARTUP_TIMEOUT_S}s. "
        f"The model may still be loading -- try increasing SERVER_STARTUP_TIMEOUT_S in config.py."
    )


def call_model(tier: str, prompt: str, system: str = None, max_tokens: int = 400) -> dict:
    _ensure_server_running(tier)
    port = MODELS[tier]["port"]

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=300,
    )
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    tokens_per_sec = out_tok / latency if latency > 0 else 0.0

    return {
        "text": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_s": latency,
        "tokens_per_sec": tokens_per_sec,
        "cost_usd": _cost(tier, in_tok, out_tok),
        "tier": tier,
        "model_id": MODELS[tier]["label"],
    }


def preload_all():
    """Start servers for all 4 tiers up front so mid-benchmark tier
    switches don't pay a cold-start cost. Each server holds its model
    in RAM once started -- make sure you have enough RAM for all 4
    (roughly 1+2+2.5+5 = ~10.5GB for these quantized models)."""
    for tier in MODELS:
        _ensure_server_running(tier)
