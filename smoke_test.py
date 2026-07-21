
import os
from config import MODEL_DIR, MODELS, LLAMA_SERVER_EXE

print(f"MODEL_DIR is set to: {MODEL_DIR}")
print(f"LLAMA_SERVER_EXE is set to: {LLAMA_SERVER_EXE}\n")

if not os.path.exists(LLAMA_SERVER_EXE):
    print(f"[MISSING] llama-server.exe not found at that path.")
    print("Find it with:")
    print(r'  dir "<your llama-b9606 folder>" -Recurse -Filter "llama-server.exe"')
    print("Then update LLAMA_SERVER_EXE in config.py (or set the env var) to the exact path shown.")
    exit(1)
else:
    print("[FOUND] llama-server.exe\n")

print("Checking that all 4 model files exist in MODEL_DIR...\n")

all_found = True
for tier, info in MODELS.items():
    path = os.path.join(MODEL_DIR, info["file"])
    exists = os.path.exists(path)
    status = "FOUND" if exists else "MISSING"
    print(f"  [{status}] {tier:6s} -> {path}")
    if not exists:
        all_found = False

if not all_found:
    print("\nOne or more files are missing. Fix MODEL_DIR in config.py "
          "(or the MODEL_DIR env var) and re-run this script.")
    exit(1)

print("\nAll files found. Starting the 'tiny' model server and asking it a test question...")
print("(first start can take 10-60s while the model loads into RAM)\n")

from models import call_model

result = call_model("tiny", "What is the capital of France? Answer in one sentence.")

print("=" * 60)
print("RESPONSE:", result["text"])
print("=" * 60)
print(f"Latency: {result['latency_s']:.2f}s | Tokens/sec: {result['tokens_per_sec']:.1f} | "
      f"Input tokens: {result['input_tokens']} | Output tokens: {result['output_tokens']}")
print("\nIf you see a sensible answer above, everything is wired up correctly.")
print("You're ready to run: python benchmark.py")
