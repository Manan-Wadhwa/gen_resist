import os
import json
import torch
import numpy as np
from Bio import SeqIO
from tkinter import Tk, filedialog

# ==========================================
# CONFIG
# ==========================================

MODEL_DIR = "model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("🧬 AMR PREDICTION – INFERENCE v0 (PROTOTYPE)")
print("=" * 60)
print(f"🖥️ Device: {DEVICE}")

# ==========================================
# LOAD METADATA
# ==========================================

def load_metadata():
    with open(os.path.join(MODEL_DIR, "antibiotics.json")) as f:
        antibiotics = json.load(f)
    return antibiotics

ANTIBIOTICS = load_metadata()
print(f"✅ Loaded {len(ANTIBIOTICS)} antibiotics\n")

# ==========================================
# SIMPLE MODEL (PLACEHOLDER)
# ==========================================

class AMRBaselineModel(torch.nn.Module):
    """
    Lightweight baseline model.
    Will be replaced with GNN-based architecture.
    """
    def __init__(self, num_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_outputs)
        )

    def forward(self, x):
        return self.net(x)

model = AMRBaselineModel(len(ANTIBIOTICS)).to(DEVICE)
model.eval()

print("⚠️ Using baseline model (GNN pending)\n")

# ==========================================
# FEATURE EXTRACTION (BASIC)
# ==========================================

def compute_genome_features(sequence):
    length = len(sequence)
    gc = (sequence.count("G") + sequence.count("C")) / max(length, 1)
    n_frac = sequence.count("N") / max(length, 1)

    return np.array([gc, length / 5_000_000, n_frac], dtype=np.float32)

def load_genome(path):
    seq = ""
    for record in SeqIO.parse(path, "fasta"):
        seq += str(record.seq).upper()
    if not seq:
        raise ValueError("Empty genome file")
    return seq

# ==========================================
# INFERENCE
# ==========================================

def predict(genome_path):
    print(f"\n🧬 Analyzing: {os.path.basename(genome_path)}")

    seq = load_genome(genome_path)
    features = compute_genome_features(seq)

    x = torch.tensor(features).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs

# ==========================================
# REPORT
# ==========================================

def generate_report(probs):
    print("\n📊 PREDICTION SUMMARY")
    print("-" * 40)

    resistant = 0
    for ab, p in zip(ANTIBIOTICS, probs):
        label = "Resistant" if p > 0.5 else "Susceptible"
        if label == "Resistant":
            resistant += 1
        print(f"{ab:20s} : {label} ({p:.2f})")

    print(f"\n📈 Resistant antibiotics: {resistant}/{len(ANTIBIOTICS)}")

# ==========================================
# FILE PICKER
# ==========================================

def select_file():
    Tk().withdraw()
    return filedialog.askopenfilename(
        title="Select Genome FASTA File",
        filetypes=[("FASTA files", "*.fna *.fasta")]
    )

# ==========================================
# MAIN
# ==========================================

def main():
    path = select_file()
    if not path:
        print("❌ No file selected")
        return

    probs = predict(path)
    generate_report(probs)

    print("\n⚠️ Note: This is an early inference prototype.")
    print("   Graph-based modeling & gene detection coming next.")

if __name__ == "__main__":
    main()
