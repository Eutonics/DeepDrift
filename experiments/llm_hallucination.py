import torch
import torch.nn as nn
import argparse
import os

# Import DeepDrift
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deepdrift.llm import DeepDriftGuard

class MockLLM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Mocking an encoder layer that returns [B, Seq, Dim]
        self.encoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.encoder:
            x = layer(x)
        return x

def run_experiment(quick=False):
    print(f"[*] Running LLM Hallucination Detection. Quick mode: {quick}")

    model = MockLLM()
    model.eval()

    # In LLMs, we usually monitor the last encoder/decoder layer
    # We'll use the last linear layer as target
    layer_names = ['encoder.2']
    monitor = DeepDriftGuard(model, threshold=0.05)
    # Manually set layer names since heuristic might fail on mock
    monitor.monitor.layer_names = layer_names
    monitor.monitor._register_all_hooks()

    if quick:
        print("[!] Generating tokens...")
        # Step 0: Context
        monitor.reset()
        input_ids = torch.tensor([[1, 2, 3]])
        _ = monitor(input_ids)

        # Step 1: Normal continuation
        next_token = torch.tensor([[4]])
        diag1 = monitor(next_token)
        print(f"Token 1: {diag1}")

        # Step 2: Hallucination (sudden shift in semantic velocity)
        # We simulate this by perturbing the model state or inputs
        # Here we just pass a very different token that would trigger a shift if the model was real
        # For mock, we'll manually perturb the prev_state to simulate a jump
        monitor.monitor.prev_state = monitor.monitor.prev_state * 10
        halluc_token = torch.tensor([[999]])
        diag2 = monitor(halluc_token)
        print(f"Token 2: {diag2}")

        if diag2.is_anomaly:
            print("✅ SUCCESS: Hallucination detected via Semantic Velocity jump!")
    else:
        print("[!] Full mode: Integration with HuggingFace (requires transformers).")
        print("    Example: monitor = DeepDriftGuard(hf_model)")
        print("    for token in model.generate(...): monitor(token)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    run_experiment(quick=args.quick)
