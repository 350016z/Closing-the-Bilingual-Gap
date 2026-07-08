import os
import subprocess

MODEL_NAME = "gpt-5.1_align_e5"

# Step 1: Convert JSON/JSONL to TSV
print("=== Step 1: Converting JSONL to TSV ===")
subprocess.run([
    "python", "convert_to_tsv.py",
    "--model_name", MODEL_NAME
])

# Step 2: Run MQM scoring
print("\n=== Step 2: Running MQM Scoring ===")
subprocess.run([
    "python", "-m", "mt_metrics_eval.converters.score_mqm",
    "--weights", "critical:25 major:5 minor:1 No-error:0",
    "--input", f"model_tsv/{MODEL_NAME}.tsv",
    "--output", f"mqm_score/{MODEL_NAME}.score",
    "--noforce_contiguous"
])

# Step 3: Compute correlations
print("\n=== Step 3: Calculating Correlation ===")
subprocess.run([
    "python", "score.py",
    "--file2", MODEL_NAME
])

print("\n✅ All steps completed successfully!")
