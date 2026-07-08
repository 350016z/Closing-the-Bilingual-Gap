#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SP / MP / SR / MR / F1 calculator (character level by default; token level optional)
SP  = | P(E) ∩ P(Ê) | / | P(Ê) |                 (precision over positions)
SR  = | P(E) ∩ P(Ê) | / | P(E) |                 (recall over positions)
SF1 = 2 * SP * SR / (SP + SR)

MP/MR/MF1 are the same metrics computed over "major"-severity spans only.

Where:
- E      : gold (human) error spans
- Ê      : predicted error spans
- E_maj  : subset of E with "major" severities
- Ê_maj  : subset of Ê with "major" severities

Data format: JSON/JSONL; each record must contain ("id","system","mt"), and mt must match between gold and pred.
Indices are 0-based, end-exclusive (character level).

Usage example:
python sp_mp.py \
--gold gold.json \
--pred pred_output/gpt-5.1_align_e5.jsonl \
--unit char \
--major-severities "Major" \
--group-by-system \
> sp_mp_output/gpt-5.1_align_e5.log
"""

from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

# ---------- Data schema ----------

@dataclass
class Span:
    start: int
    end: int
    severity: Optional[str] = None
    type: Optional[str] = None
    text: Optional[str] = None

@dataclass
class Example:
    id: str
    system: str
    src: str
    mt: str
    ref: Optional[str]
    spans: List[Span]
    token_offsets: Optional[List[Tuple[int, int]]] = None  # optional, for token-level

# ---------- Utilities : loader: JSON or JSONL ----------

def load_json_any(path: str) -> List[dict]:
    """
    Accept:
      - JSON array: [ {...}, {...} ]
      - Single JSON object: { ... }  (wrapped into a list)
      - JSONL: one JSON object per line
    """
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    if not data:
        return []

    first = data.lstrip()[:1]
    if first in ("[", "{"):
        try:
            obj = json.loads(data)
            if isinstance(obj, list):
                return obj
            elif isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass  # fall back to JSONL

    out = []
    for i, line in enumerate(data.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL parse error at line {i}: {e}") from e
    return out

def to_example(obj: dict) -> Example:
    id_ = obj.get("id")
    system = obj.get("system", "default")
    src = obj.get("src", "")
    if "mt" not in obj:
        raise ValueError(f"Missing 'mt' in example id={id_}, system={system}")
    mt = obj["mt"]
    ref = obj.get("ref", "")

    # Spans: accept several key names; empty arrays are allowed
    spans_in = obj.get("spans")
    if spans_in is None:
        spans_in = obj.get("annotations", obj.get("errors", []))
    if spans_in is None:
        spans_in = []

    spans: List[Span] = []
    for s in spans_in:
        start = s.get("start_index", s.get("start", s.get("begin")))
        end = s.get("end_index", s.get("end", s.get("finish")))
        if start is None or end is None:
            continue
        start = int(start); end = int(end)
        if end <= start:
            continue
        text = s.get("error_text_segment", s.get("error_text", s.get("text")))
        etype = s.get("error_type", s.get("type"))
        severity = s.get("error_severity", s.get("severity"))
        spans.append(Span(start=start, end=end, severity=severity, type=etype, text=text))

    tok_offs = obj.get("token_offsets")
    if tok_offs is not None:
        tok_offs = [(int(a), int(b)) for a, b in tok_offs]

    if id_ is None:
        id_ = obj.get("_id") or obj.get("seg_id") or obj.get("doc_id") or f"line_{hash(src + mt) & 0xFFFFFFFF}"

    return Example(id=id_, system=system, src=src, mt=mt, ref=ref, spans=spans, token_offsets=tok_offs)

# ---------- Position set builders ----------

def clip(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))

def span_to_char_positions(start: int, end: int, length: int) -> Set[int]:
    # 0-based, end-exclusive; clamp to [0, length]
    s = clip(start, 0, length)
    e = clip(end,   0, length)
    if e <= s:
        return set()
    return set(range(s, e))

def span_to_token_positions(start: int, end: int, token_offsets: List[Tuple[int, int]]) -> Set[int]:
    # Map char span to token indices: include token t if [tok_s, tok_e) intersects [start,end)
    positions = set()
    for t_idx, (tok_s, tok_e) in enumerate(token_offsets):
        if tok_e <= start or end <= tok_s:
            continue
        positions.add(t_idx)
    return positions

def build_position_set(
    spans: List[Span],
    length: int,
    unit: str = "char",
    token_offsets: Optional[List[Tuple[int, int]]] = None,
    major_severities: Optional[Set[str]] = None,
    only_major: bool = False,
) -> Set[int]:
    maj_lower = {s.lower() for s in (major_severities or set())}
    pos: Set[int] = set()
    for sp in spans:
        if only_major and maj_lower:
            sev = (sp.severity or "").lower()
            if sev not in maj_lower:
                continue
        if unit == "char":
            pos |= span_to_char_positions(sp.start, sp.end, length)
        elif unit == "token":
            if token_offsets is None:
                raise ValueError("token_offsets is required for unit='token'")
            pos |= span_to_token_positions(sp.start, sp.end, token_offsets)
        else:
            raise ValueError(f"Unknown unit: {unit}")
    return pos

# ---------- SP / SR / F1 computation ----------

@dataclass
class SPMP:
    sp: Optional[float]          # precision over positions
    mp: Optional[float]          # precision over positions (major)
    sr: Optional[float]          # recall over positions
    mr: Optional[float]          # recall over positions (major)
    sf1: Optional[float]         # F1 over positions
    mf1: Optional[float]         # F1 over positions (major)
    num_pred_pos: int
    num_pred_major_pos: int
    num_gold_pos: int
    num_gold_major_pos: int
    overlap: int
    overlap_major: int

def f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)

def compute_sp_mp_for_pair(
    gold: Example,
    pred: Example,
    unit: str = "char",
    major_severities: Optional[Set[str]] = None,
) -> SPMP:
    if gold.mt != pred.mt:
        raise ValueError(f"MT text mismatch for id={gold.id}, system={gold.system}")

    # If major_severities is not provided, use a sensible default
    maj_set = {s.lower() for s in (major_severities or {"major", "critical", "severe"})}

    length = len(gold.mt)
    gold_pos = build_position_set(gold.spans, length, unit, gold.token_offsets, maj_set, only_major=False)
    gold_pos_maj = build_position_set(gold.spans, length, unit, gold.token_offsets, maj_set, only_major=True)

    pred_pos = build_position_set(pred.spans, length, unit, pred.token_offsets, maj_set, only_major=False)
    pred_pos_maj = build_position_set(pred.spans, length, unit, pred.token_offsets, maj_set, only_major=True)

    inter = len(gold_pos & pred_pos)
    inter_maj = len(gold_pos_maj & pred_pos_maj)

    sp = (inter / len(pred_pos)) if len(pred_pos) > 0 else None
    sr = (inter / len(gold_pos)) if len(gold_pos) > 0 else None
    sf1 = f1(sp, sr)

    mp = (inter_maj / len(pred_pos_maj)) if len(pred_pos_maj) > 0 else None
    mr = (inter_maj / len(gold_pos_maj)) if len(gold_pos_maj) > 0 else None
    mf1 = f1(mp, mr)

    return SPMP(
        sp=sp,
        mp=mp,
        sr=sr,
        mr=mr,
        sf1=sf1,
        mf1=mf1,
        num_pred_pos=len(pred_pos),
        num_pred_major_pos=len(pred_pos_maj),
        num_gold_pos=len(gold_pos),
        num_gold_major_pos=len(gold_pos_maj),
        overlap=inter,
        overlap_major=inter_maj,
    )

# ---------- Aggregation ----------

@dataclass
class Aggregates:
    micro_sp: Optional[float]
    micro_sr: Optional[float]
    micro_sf1: Optional[float]
    micro_mp: Optional[float]
    micro_mr: Optional[float]
    micro_mf1: Optional[float]
    macro_sp: Optional[float]
    macro_sr: Optional[float]
    macro_sf1: Optional[float]
    macro_mp: Optional[float]
    macro_mr: Optional[float]
    macro_mf1: Optional[float]
    total_pred_pos: int
    total_pred_major_pos: int
    total_gold_pos: int
    total_gold_major_pos: int
    total_overlap: int
    total_overlap_major: int
    n_items_with_pred: int
    n_items_with_pred_major: int
    n_items_with_gold: int
    n_items_with_gold_major: int

def aggregate(results: List[SPMP]) -> Aggregates:
    total_pred = sum(r.num_pred_pos for r in results)
    total_gold = sum(r.num_gold_pos for r in results)
    total_overlap = sum(r.overlap for r in results)

    micro_sp = (total_overlap / total_pred) if total_pred > 0 else None
    micro_sr = (total_overlap / total_gold) if total_gold > 0 else None
    micro_sf1 = f1(micro_sp, micro_sr)

    total_pred_major = sum(r.num_pred_major_pos for r in results)
    total_gold_major = sum(r.num_gold_major_pos for r in results)
    total_overlap_major = sum(r.overlap_major for r in results)

    micro_mp = (total_overlap_major / total_pred_major) if total_pred_major > 0 else None
    micro_mr = (total_overlap_major / total_gold_major) if total_gold_major > 0 else None
    micro_mf1 = f1(micro_mp, micro_mr)

    sp_vals = [r.sp for r in results if r.sp is not None]
    sr_vals = [r.sr for r in results if r.sr is not None]
    sf1_vals = [r.sf1 for r in results if r.sf1 is not None]

    mp_vals = [r.mp for r in results if r.mp is not None]
    mr_vals = [r.mr for r in results if r.mr is not None]
    mf1_vals = [r.mf1 for r in results if r.mf1 is not None]

    macro_sp = (sum(sp_vals) / len(sp_vals)) if sp_vals else None
    macro_sr = (sum(sr_vals) / len(sr_vals)) if sr_vals else None
    macro_sf1 = (sum(sf1_vals) / len(sf1_vals)) if sf1_vals else None

    macro_mp = (sum(mp_vals) / len(mp_vals)) if mp_vals else None
    macro_mr = (sum(mr_vals) / len(mr_vals)) if mr_vals else None
    macro_mf1 = (sum(mf1_vals) / len(mf1_vals)) if mf1_vals else None

    return Aggregates(
        micro_sp=micro_sp, micro_sr=micro_sr, micro_sf1=micro_sf1,
        micro_mp=micro_mp, micro_mr=micro_mr, micro_mf1=micro_mf1,
        macro_sp=macro_sp, macro_sr=macro_sr, macro_sf1=macro_sf1,
        macro_mp=macro_mp, macro_mr=macro_mr, macro_mf1=macro_mf1,
        total_pred_pos=total_pred,
        total_pred_major_pos=total_pred_major,
        total_gold_pos=total_gold,
        total_gold_major_pos=total_gold_major,
        total_overlap=total_overlap,
        total_overlap_major=total_overlap_major,
        n_items_with_pred=len(sp_vals),
        n_items_with_pred_major=len(mp_vals),
        n_items_with_gold=len(sr_vals),
        n_items_with_gold_major=len(mr_vals),
    )

# ---------- IO & Matching ----------

Key = Tuple[str, str]  # (id, system)

def index_examples(objs: List[dict]) -> Dict[Key, Example]:
    out: Dict[Key, Example] = {}
    for obj in objs:
        ex = to_example(obj)
        key = (ex.id, ex.system)
        if key in out:
            raise ValueError(f"Duplicate key: {key}")
        out[key] = ex
    return out

def evaluate(gold_path: str, pred_path: str, unit: str, major_severities: List[str], group_by_system: bool):
    gold_raw = load_json_any(gold_path)
    pred_raw = load_json_any(pred_path)

    gold_ix = index_examples(gold_raw)
    pred_ix = index_examples(pred_raw)

    keys = sorted(set(gold_ix.keys()) & set(pred_ix.keys()))
    if not keys:
        raise ValueError("No overlapping (id, system) keys between gold and pred.")

    ###################################################################
    # Optionally set ids to skip (as ints); non-numeric ids are ignored by this filter
    skip_ids = {2, 3, 7, 8, 9, 50, 59, 60, 1007, 1025, 1037} 
    def to_int_or_none(x):
        try:
            return int(x)
        except:
            return None
    keys = [k for k in keys if to_int_or_none(k[0]) not in skip_ids]
    ###################################################################

    maj_set = set(major_severities)

    # Per-item results
    results: Dict[Key, SPMP] = {}
    for key in keys:
        res = compute_sp_mp_for_pair(gold_ix[key], pred_ix[key], unit=unit, major_severities=maj_set)
        results[key] = res

    # Aggregate overall
    agg = aggregate(list(results.values()))

    def fmt(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    print("== Overall (position-level) ==")
    print(f"Micro  SP={fmt(agg.micro_sp)}  SR={fmt(agg.micro_sr)}  SF1={fmt(agg.micro_sf1)}"
          f"   (overlap={agg.total_overlap}, pred_pos={agg.total_pred_pos}, gold_pos={agg.total_gold_pos})")
    print(f"Micro* MP={fmt(agg.micro_mp)}  MR={fmt(agg.micro_mr)}  MF1={fmt(agg.micro_mf1)}"
          f"   (overlap_major={agg.total_overlap_major}, pred_major_pos={agg.total_pred_major_pos}, gold_major_pos={agg.total_gold_major_pos})")
    print(f"Macro  SP={fmt(agg.macro_sp)}  SR={fmt(agg.macro_sr)}  SF1={fmt(agg.macro_sf1)}"
          f"   (items_with_pred={agg.n_items_with_pred}, items_with_gold={agg.n_items_with_gold})")
    print(f"Macro* MP={fmt(agg.macro_mp)}  MR={fmt(agg.macro_mr)}  MF1={fmt(agg.macro_mf1)}"
          f"   (items_with_pred_major={agg.n_items_with_pred_major}, items_with_gold_major={agg.n_items_with_gold_major})")

    # Per-system aggregates (optional)
    if group_by_system:
        print("\n== Per-system (Micro/Macro) ==")