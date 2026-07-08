#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stratified Nested Repeated K-Fold evaluation for Taiwanese translation error span detection.

Example:

python stratified_nested_rkfold_eval.py \
  --gold gold.json \
  --preds \
  "A=pred_output/pred_1.jsonl, \
  B=pred_output/pred_2.jsonl, \
  C=pred_output/pred_3.jsonl" \
  --outdir snrkf_out/example \
  --outer-k 5 \
  --outer-repeats 10 \
  --inner-k 4 \
  --seed 42 \
  --unit char \
  --major-severities "Major" \
  --select-metric micro_sf1

Notes:
- Stratification label is derived from gold severity:
  severity == "No-error" -> y=0 (no error)
  otherwise -> y=1
- Outer loop: RepeatedStratifiedKFold
- Inner loop: StratifiedKFold (shuffle) on outer-train, selects best candidate by mean inner metric
- Outer test: computes full metrics (SP/MP micro + MQM correlations) like your original script.
"""

import os
import json
import math
import argparse
import random
import subprocess
from typing import Dict, List, Tuple, Any, Optional

import trans_taigi.sp_mp as sp
import convert_to_tsv as c2t
from score import load_scores_with_id

from scipy.stats import pearsonr, spearmanr, kendalltau, t

try:
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
except Exception as e:
    raise ImportError(
        "This script requires scikit-learn. Please install it, e.g. pip install scikit-learn"
    ) from e


Key = Tuple[str, str]  # (id, system)

SKIP_IDS = {2, 3, 7, 8, 9, 50, 59, 60, 1007, 1025, 1037}


# -------------------------
# Utilities (mostly reused)
# -------------------------
def _to_int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def load_json_any(path: str) -> List[dict]:
    return sp.load_json_any(path)


def index_objs_by_key(objs: List[dict]) -> Dict[Key, dict]:
    out: Dict[Key, dict] = {}
    for obj in objs:
        ex = sp.to_example(obj)
        key = (ex.id, ex.system)
        if key in out:
            raise ValueError(f"Duplicate key: {key}")
        out[key] = obj
    return out


def filter_keys(keys: List[Key]) -> List[Key]:
    kept = []
    for (id_, sysname) in keys:
        iid = _to_int_or_none(id_)
        if iid is not None and iid in SKIP_IDS:
            continue
        kept.append((id_, sysname))
    return kept


def write_jsonl(objs: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_corr(x: List[float], y: List[float]) -> Dict[str, Optional[float]]:
    if len(x) < 2 or len(y) < 2:
        return {"pearson": None, "spearman": None, "kendall": None}
    if len(set(x)) == 1 or len(set(y)) == 1:
        return {"pearson": None, "spearman": None, "kendall": None}

    pr, _ = pearsonr(x, y)
    sr, _ = spearmanr(x, y)
    kr, _ = kendalltau(x, y)
    return {"pearson": float(pr), "spearman": float(sr), "kendall": float(kr)}


def spmp_micro_metrics_from_index(
    gold_ix: Dict[Key, Any],
    pred_ix: Dict[Key, Any],
    keys: List[Key],
    unit: str,
    major_severities: List[str],
) -> Dict[str, Optional[float]]:
    """
    Fast in-memory micro SP/MP metrics on a subset of keys.
    """
    maj_set = set(major_severities)
    per_item = []
    for key in keys:
        if key not in gold_ix or key not in pred_ix:
            continue
        per_item.append(
            sp.compute_sp_mp_for_pair(
                gold_ix[key],
                pred_ix[key],
                unit=unit,
                major_severities=maj_set,
            )
        )

    if not per_item:
        return {
            "micro_sp": None, "micro_sr": None, "micro_sf1": None,
            "micro_mp": None, "micro_mr": None, "micro_mf1": None,
            "n_items": 0,
        }

    agg = sp.aggregate(per_item)
    return {
        "micro_sp": agg.micro_sp,
        "micro_sr": agg.micro_sr,
        "micro_sf1": agg.micro_sf1,
        "micro_mp": agg.micro_mp,
        "micro_mr": agg.micro_mr,
        "micro_mf1": agg.micro_mf1,
        "n_items": len(per_item),
    }


def mqm_score_file_from_jsonl(jsonl_path: str, tsv_path: str, score_path: str, weights: str) -> None:
    c2t.convert_to_tsv(jsonl_path, tsv_path)
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    subprocess.run(
        [
            "python", "-m", "mt_metrics_eval.converters.score_mqm",
            "--weights", weights,
            "--input", tsv_path,
            "--output", score_path,
            "--noforce_contiguous",
        ],
        check=True
    )


def mean_std_ci95(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]], int]:
    """
    mean ± std and 95% CI using Student-t (better for small n).
    """
    clean = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        clean.append(float(v))

    n = len(clean)
    if n == 0:
        return None, None, None, 0
    if n == 1:
        return clean[0], 0.0, (clean[0], clean[0]), 1

    m = sum(clean) / n
    var = sum((v - m) ** 2 for v in clean) / (n - 1)
    s = math.sqrt(var)

    # t critical for 95% CI
    alpha = 0.05
    tcrit = float(t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * (s / math.sqrt(n))
    return m, s, (m - half, m + half), n


# -------------------------
# Stratification label
# -------------------------
def extract_no_error_label_from_gold_obj(obj: dict) -> int:
    """
    Returns y in {0,1} for stratification:
    - y=0 if severity == "No-error" (no translation error)
    - y=1 otherwise

    Robust handling:
    - If top-level "severity" exists and is str: use it.
    - Else, try to find severity in lists under common fields.
    - If nothing found, default to y=1 (conservative) and warn upstream (counted).
    """
    sev = obj.get("severity", None)
    if isinstance(sev, str):
        return 0 if sev.strip() == "No-error" else 1

    # Try common list containers (customize if your schema differs)
    candidates = []
    for k in ["errors", "annotations", "spans", "edits"]:
        v = obj.get(k)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict) and isinstance(it.get("severity"), str):
                    candidates.append(it["severity"].strip())

    if candidates:
        # If ALL are No-error => no-error, else error
        all_no_error = all(s == "No-error" for s in candidates)
        return 0 if all_no_error else 1

    # Unknown schema => be conservative
    return 1


def parse_preds_arg(preds: str) -> Dict[str, str]:
    """
    Parse --preds "name=path,name2=path2"
    """
    out: Dict[str, str] = {}
    parts = [p.strip() for p in preds.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid --preds item: {p}. Expected name=path")
        name, path = p.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Empty candidate name in --preds: {p}")
        if not path:
            raise ValueError(f"Empty candidate path in --preds: {p}")
        if name in out:
            raise ValueError(f"Duplicate candidate name: {name}")
        out[name] = path
    if not out:
        raise ValueError("--preds parsed empty. Provide at least one candidate.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold JSON/JSONL（全資料）")

    # Candidates
    ap.add_argument("--preds", required=True,
                    help='候選系統/參數的 pred 檔：用逗號分隔 name=path，例如 "A=a.jsonl,B=b.jsonl"')

    ap.add_argument("--outdir", default="snrkf_out")

    # Outer: repeated stratified k-fold
    ap.add_argument("--outer-k", type=int, default=5)
    ap.add_argument("--outer-repeats", type=int, default=10)

    # Inner: stratified k-fold on outer-train
    ap.add_argument("--inner-k", type=int, default=4)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--unit", choices=["char", "token"], default="char")
    ap.add_argument("--major-severities", default="Major", help="例如 'Major,Critical'")
    ap.add_argument("--mqm-weights", default="critical:25 major:5 minor:1 No-error:0")

    ap.add_argument("--select-metric", default="micro_sf1",
                    choices=["micro_sp", "micro_sr", "micro_sf1", "micro_mp", "micro_mr", "micro_mf1"],
                    help="inner loop 用來選最佳候選的指標（預設 micro_sf1）")

    args = ap.parse_args()

    majors = [s.strip() for s in args.major_severities.split(",") if s.strip()]
    cand_paths = parse_preds_arg(args.preds)
    cand_names = sorted(cand_paths.keys())

    # ---- Load gold
    gold_all = load_json_any(args.gold)
    gold_by = index_objs_by_key(gold_all)

    # ---- Load candidates preds
    pred_by_cand: Dict[str, Dict[Key, dict]] = {}
    for name in cand_names:
        pred_all = load_json_any(cand_paths[name])
        pred_by_cand[name] = index_objs_by_key(pred_all)

    # ---- Determine common keys across all candidates + gold
    keys = set(gold_by.keys())
    for name in cand_names:
        keys = keys & set(pred_by_cand[name].keys())
    keys = sorted(filter_keys(list(keys)))

    if len(keys) < args.outer_k:
        raise ValueError(f"After filtering, keys={len(keys)} < outer_k={args.outer_k}")

    # ---- Build stratification labels y from gold severity
    unknown_schema = 0
    y = []
    for k in keys:
        lab = extract_no_error_label_from_gold_obj(gold_by[k])
        # If schema unknown => function returns 1 (conservative).
        # We can't perfectly detect "unknown" here without extra signals,
        # so just keep a simple heuristic: if no 'severity' key and no list found, count as unknown-ish.
        if "severity" not in gold_by[k]:
            # rough count; still fine
            unknown_schema += 1
        y.append(lab)

    if unknown_schema > 0:
        print(f"[WARN] {unknown_schema} gold items have no top-level 'severity'. "
              f"Stratification label will be inferred (or defaulted conservatively).")

    # ---- Pre-index examples for SP/MP fast computation (gold once + pred per candidate)
    gold_ix = sp.index_examples([gold_by[k] for k in keys])

    pred_ix_cand: Dict[str, Dict[Key, Any]] = {}
    for name in cand_names:
        pred_ix_cand[name] = sp.index_examples([pred_by_cand[name][k] for k in keys])

    # ---- Outer split: RepeatedStratifiedKFold
    os.makedirs(args.outdir, exist_ok=True)
    outer = RepeatedStratifiedKFold(
        n_splits=args.outer_k,
        n_repeats=args.outer_repeats,
        random_state=args.seed
    )

    rows: List[Dict[str, Any]] = []

    total_outer = args.outer_k * args.outer_repeats
    for outer_iter, (train_idx, test_idx) in enumerate(outer.split(keys, y)):
        rep = outer_iter // args.outer_k
        fold = outer_iter % args.outer_k

        train_keys = [keys[i] for i in train_idx]
        test_keys = [keys[i] for i in test_idx]

        y_train = [y[i] for i in train_idx]

        # ---- Inner split on outer-train: StratifiedKFold
        inner = StratifiedKFold(
            n_splits=args.inner_k,
            shuffle=True,
            random_state=args.seed + 1000 * rep + fold
        )

        # ---- Evaluate each candidate on inner CV (fast SP/MP only) and pick best
        cand_inner_scores: Dict[str, List[Optional[float]]] = {name: [] for name in cand_names}

        for i_iter, (in_tr_idx, in_val_idx) in enumerate(inner.split(train_keys, y_train)):
            val_keys = [train_keys[i] for i in in_val_idx]

            for name in cand_names:
                m = spmp_micro_metrics_from_index(
                    gold_ix=gold_ix,
                    pred_ix=pred_ix_cand[name],
                    keys=val_keys,
                    unit=args.unit,
                    major_severities=majors,
                )
                cand_inner_scores[name].append(m.get(args.select_metric))

        # mean inner score for selection
        def _mean(xs: List[Optional[float]]) -> float:
            clean = [float(v) for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return sum(clean) / max(1, len(clean))

        inner_means = {name: _mean(cand_inner_scores[name]) for name in cand_names}
        # pick best (tie-break: higher mean, then name)
        best_name = sorted(cand_names, key=lambda n: (-inner_means[n], n))[0]

        # ---- Outer evaluation (full metrics like original) on best candidate
        fold_dir = os.path.join(args.outdir, f"rep_{rep:02d}", f"fold_{fold:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        gold_test_path = os.path.join(fold_dir, "gold_test.jsonl")
        pred_test_path = os.path.join(fold_dir, f"pred_test__{best_name}.jsonl")

        gold_test_objs = [gold_by[k] for k in test_keys]
        pred_test_objs = [pred_by_cand[best_name][k] for k in test_keys]

        write_jsonl(gold_test_objs, gold_test_path)
        write_jsonl(pred_test_objs, pred_test_path)

        # (A) micro SP/MP on outer test
        pred_test_ix = sp.index_examples(pred_test_objs)
        m_outer = spmp_micro_metrics_from_index(
            gold_ix=gold_ix,
            pred_ix=pred_test_ix,  # this ix is only for test subset; simplest
            keys=test_keys,
            unit=args.unit,
            major_severities=majors,
        )

        # (B) MQM score + correlations on outer test (same as your original approach)
        gold_tsv = os.path.join(fold_dir, "gold.tsv")
        pred_tsv = os.path.join(fold_dir, f"pred__{best_name}.tsv")
        gold_score = os.path.join(fold_dir, "gold.score")
        pred_score = os.path.join(fold_dir, f"pred__{best_name}.score")

        mqm_score_file_from_jsonl(gold_test_path, gold_tsv, gold_score, weights=args.mqm_weights)
        mqm_score_file_from_jsonl(pred_test_path, pred_tsv, pred_score, weights=args.mqm_weights)

        s_gold = load_scores_with_id(gold_score)
        s_pred = load_scores_with_id(pred_score)

        common_ids = sorted(set(s_gold.keys()) & set(s_pred.keys()))
        x = [s_gold[i] for i in common_ids]
        y_ = [s_pred[i] for i in common_ids]
        corr = safe_corr(x, y_)

        row = {
            "repeat": rep,
            "fold": fold,
            "outer_iter": outer_iter,
            "n_test": m_outer["n_items"],

            "best_candidate": best_name,
            f"inner_mean_{args.select_metric}": inner_means[best_name],

            "micro_sp": m_outer["micro_sp"],
            "micro_sr": m_outer["micro_sr"],
            "micro_sf1": m_outer["micro_sf1"],
            "micro_mp": m_outer["micro_mp"],
            "micro_mr": m_outer["micro_mr"],
            "micro_mf1": m_outer["micro_mf1"],

            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "kendall": corr["kendall"],
        }
        rows.append(row)

        print(
            f"[outer {outer_iter+1:03d}/{total_outer} | rep={rep:02d} fold={fold:02d}] "
            f"best={best_name} inner_{args.select_metric}={inner_means[best_name]:.4f} "
            f"micro_sf1={m_outer['micro_sf1'] if m_outer['micro_sf1'] is not None else 'n/a'} "
            f"pearson={corr['pearson'] if corr['pearson'] is not None else 'n/a'}"
        )

    # ======================
    # Summary over OUTER tests
    # ======================
    summary_cols = [
        "micro_sp", "micro_sr", "micro_sf1",
        "micro_mp", "micro_mr", "micro_mf1",
        "pearson", "spearman", "kendall",
    ]

    summary_lines = []
    summary_json = {
        "outer_k": args.outer_k,
        "outer_repeats": args.outer_repeats,
        "inner_k": args.inner_k,
        "seed": args.seed,
        "unit": args.unit,
        "select_metric": args.select_metric,
        "n_total_outer_folds": len(rows),
        "candidates": cand_names,
        "metrics": {},
        "selection_frequency": {},
    }

    # selection frequency
    sel = {}
    for r in rows:
        sel[r["best_candidate"]] = sel.get(r["best_candidate"], 0) + 1
    summary_json["selection_frequency"] = {k: sel.get(k, 0) for k in cand_names}

    summary_lines.append("== Summary over OUTER test folds (nested CV) ==")
    summary_lines.append(f"outer: {args.outer_k}-fold x {args.outer_repeats} repeats  (total={len(rows)})")
    summary_lines.append(f"inner: {args.inner_k}-fold stratified, select_metric={args.select_metric}")
    summary_lines.append("== Best-candidate selection frequency ==")
    for name in cand_names:
        summary_lines.append(f"{name}: {sel.get(name, 0)}/{len(rows)}")

    summary_lines.append("\n== Metrics (mean ± std, 95% CI) ==")
    for c in summary_cols:
        vals = [row.get(c) for row in rows]
        mean, std, ci, n = mean_std_ci95(vals)
        if mean is None:
            summary_lines.append(f"{c}: n/a")
            summary_json["metrics"][c] = None
        else:
            # summary_lines.append(
            #     f"{c}: mean={mean:.4f} std={std:.4f}  95%CI=[{ci[0]:.4f},{ci[1]:.4f}] (n={n})"
            # )
            summary_lines.append(
                f"{c}: mean={mean:.4f} std={std:.4f}"
            )
            summary_json["metrics"][c] = {
                "mean": mean,
                "std": std,
                "ci95": [ci[0], ci[1]],
                "n": n
            }

    # ======================
    # Save per-outer CSV + Summary
    # ======================
    csv_path = os.path.join(args.outdir, "snrkf_results.csv")
    cols = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            def fmt(v):
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.6f}"
                return str(v)
            f.write(",".join(fmt(row.get(c)) for c in cols) + "\n")

    summary_txt_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    summary_json_path = os.path.join(args.outdir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print(f"\nSaved per-outer CSV: {csv_path}")
    print(f"Saved summary txt:  {summary_txt_path}")
    print(f"Saved summary json: {summary_json_path}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()