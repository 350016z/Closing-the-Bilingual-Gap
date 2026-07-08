# Closing the Bilingual Gap: Cross-lingual Semantic Scaffolding for Fine-Grained Machine Translation Evaluation in Taiwanese Hokkien

This repository accompanies our work on **translation error-span detection** for
Mandarin → Taiwanese machine translation. It provides (1) an LLM-based pipeline that
detects and scores translation-error spans using **Cross-lingual Semantic Scaffolding**,
and (2) an **AI-assisted human annotation** interface.

## Repository structure

```
Translation-Error-main/
├── trans_taigi/          # Main pipeline: prediction, scoring, evaluation
├── data/                 # Retrieval corpora (see data/README.md)
├── ai-assist_interface/  # AI-assisted human annotation tool (see its README.md)
├── requirements.txt
└── README.md
```

- **Retrieval Corpora** — sentence- and word-level Chinese↔Taiwanese references used
  to build the Alignment Hints. Full description and sources: [`data/README.md`](data/README.md).
- **AI-Assisted Human Annotation** — FastAPI backend + Next.js frontend for reviewing
  and correcting model predictions. Full description: [`ai-assist_interface/README.md`](ai-assist_interface/README.md).

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt

# External MQM scorer (used by trans_taigi/run_score.py)
pip install git+https://github.com/google-research/mt-metrics-eval.git

# API keys: copy the template and fill in your credentials
cp trans_taigi/.env.template trans_taigi/.env   # Azure OpenAI / Gemini / Qwen
```

## Pipeline (`trans_taigi/`)

1. **Predict error spans** with an LLM, optionally augmented with Alignment Hints:
   `pred_with_gemini_alignment.py`, `pred_with_gpt_alignment.py`,
   `pred_with_qwen_alignment.py` → writes to `pred_output/`.
2. **Prepare gold data**: the gold error-span annotations come from prior released
   data; `original_to_gold.py` converts them into `gold.json`.
3. **MQM scoring**: `convert_to_tsv.py` converts predictions/gold to TSV, then
   `run_score.py` runs the MQM scorer and correlation analysis (`score.py`)
   → `model_tsv/`, `mqm_score/`.
4. **Span-level metrics**: `sp_mp.py` computes SP / MP / SR / MR / F1 over predicted
   vs. gold error positions → `sp_mp_output/`.
5. **Model comparison**: `stratified_nested_rkfold_eval.py` runs stratified nested
   repeated k-fold evaluation → `snrkf_out/`.

Note: `data/`, `pred_output/`, and `snrkf_out/` ship with small example subsets only;
see the per-folder notes for how to reconstruct the full data.
