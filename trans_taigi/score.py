import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import argparse

def load_scores_with_id(filepath):
    """
    Read .score file and extract (id, score) pairs.
    Example file format:
    nllb	flores_plus	1	-1.0	qianshi2=1.0
    nllb	flores_plus	2	-7.0    qianshi2=7.0
    """
    scores = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                sent_id = int(parts[2])        # id
                score = float(parts[3])        # mqm score
                scores[sent_id] = score
            except ValueError:
                continue
    return scores


def compare_scores(file1, file2):
    scores1 = load_scores_with_id(file1)
    scores2 = load_scores_with_id(file2)

    common_ids = sorted(set(scores1.keys()) & set(scores2.keys()))
    if not common_ids:
        print("No common IDs found between the two files.")
        return

    s1 = [scores1[i] for i in common_ids]
    s2 = [scores2[i] for i in common_ids]

    print(f"## Loaded {len(common_ids)} matching IDs between the two files")

    pearson_corr, _ = pearsonr(s1, s2)
    spearman_corr, _ = spearmanr(s1, s2)
    kendall_corr, _ = kendalltau(s1, s2)

    print("\n## Correlation Results:")
    print(f"Pearson  : {pearson_corr:.4f}")
    print(f"Spearman : {spearman_corr:.4f}")
    print(f"Kendall  : {kendall_corr:.4f}")

    only_in_1 = set(scores1.keys()) - set(scores2.keys())
    only_in_2 = set(scores2.keys()) - set(scores1.keys())
    if only_in_1 or only_in_2:
        print("\n## Warning: Some IDs are not shared between files:")
        print(f" - Only in {file1}: {sorted(list(only_in_1))[:10]}{'...' if len(only_in_1) > 10 else ''}")
        print(f" - Only in {file2}: {sorted(list(only_in_2))[:10]}{'...' if len(only_in_2) > 10 else ''}")
        print(f" - Total: {len(only_in_1)}, {len(only_in_2)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare MQM scores between two .score files")
    ap.add_argument("--file1", type=str, required=False, default="gold.score", help="First .score file (e.g., human annotations)")
    ap.add_argument("--file2", type=str, required=True, default="mqm_score/gpt-4o.score", help="Second .score file (e.g., model predictions)")
    args = ap.parse_args()

    MODEL_NAME = args.file2
    file1 = f"gold.score"   # human annotator 
    file2 = f"mqm_score/{MODEL_NAME}.score"     # model prediction
    compare_scores(file1, file2)
