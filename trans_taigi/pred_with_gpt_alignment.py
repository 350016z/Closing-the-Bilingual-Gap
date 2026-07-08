import os, re, json, sys
from tqdm import tqdm
import pandas as pd
import unicodedata
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv
from json_repair import repair_json

# Retrieval / word-segmentation dependencies
import numpy as np
import jieba

load_dotenv()
from openai import AzureOpenAI
AZURE_API_KEY = os.environ['AZURE_OPENAI_API_KEY'] 
AZURE_ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT'] 
AZURE_API_VERSION = os.environ['AZURE_API_VERSION'] 
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)
MODEL_NAME = "gpt-5.2"

SYSTEM_PROMPT = """你是一個翻譯錯誤分析助手。請根據以下「原文（中文）」與「機器翻譯（台語漢字/台文）」進行錯誤偵測與**文字內嵌標記**，並以**單一 JSON 物件**輸出結果。

## 任務

1. 比對原文與機器翻譯，在「機器翻譯」文字中以 `<spanN>...</spanN>` 方式**直接標記出所有錯誤片段**（可能不只一處）。
2. 為每個被標記的錯誤片段建立一筆錯誤記錄，使用下列欄位與分類集合。
3. 為每個錯誤片段 `<spanN>...</spanN>` 給出簡短 explanation，說明為何這樣標註，並且給予建議翻譯。
4. 最後對「整個」翻譯段落給予一個0-100的整體評分。 評分標準大致如下：
	* 0：幾乎失去所有原文資訊，無法理解其意義，語法可忽略。
	* 33：翻譯保留了原文部分意思，但漏失重要資訊，語意不清，語法可能也有錯。
	* 66：大部分原文意思保留，僅有少量語法或脈絡錯誤。
	* 100：意思完全正確，與原文一致且語法也無誤。
5. 嚴格遵守下方 **RULES** 與 **輸出格式**。除了 JSON 外，不要輸出任何文字。

## 錯誤嚴重程度（severity）

* Major：扭曲關鍵事實／關係／語義，或導致讀者難以理解主要意思。
* Minor：不影響主要意思，讀者仍可大致理解，但用詞不夠精確、不自然或有其他小瑕疵（如輕微語法、標點、拼字、可讀性）。 

## 錯誤分類（error_type）

* Accuracy：是否準確傳達原意
    * Mistranslation（誤譯）
    * Addition（多譯）
    * Omission（漏譯）
* Fluency：語法與語言自然度
    * Grammar（語法錯誤）
    * Spelling（拼字錯誤）
    * Punctuation（標點誤用）
    * Inconsistency（機器翻譯內部不一致）
    * Register（語氣不合）
* Terminology：術語是否合適
    * Inappropriate（不符合領域用語）
    * Inconsistent（術語翻譯不一致）
* Style：
    * Awkward（不自然/拗口）
* Locale：
    * Currency format / Time format / Name format / Date format / Address format
* Purity（語言純正性）：例如在華語翻譯成台語時，是否混用華語詞彙，未使用台語正確表達

> 嚴禁使用上述集合以外的類別名稱。

## 標註原則

* **最小必要長度**：只包住真正錯誤的最小字串，避免將正確字詞一併包含；俗語、成語和台語固定用法則應包含整個完整內容。
* **逐字精準**：以逐字比對定位錯誤，拒絕以過長片段代替精準定位。
* **標記順序與編號**：由左至右掃描「機器翻譯」，第一次出現的錯誤為 `<span1>...</span1>`，第二個為 `<span2>...</span2>`，以此類推。
* **非重疊**：**不得巢狀**、**不得重疊** 標籤。若看似需要重疊，請拆成**相鄰且互不重疊**的最小片段。
* **文字原樣保留**：除插入 `<spanN>` 標籤外，「機器翻譯」其餘文字（含空白與標點）必須**逐字不變**。
* **Omission 規則**：
	* 在「機器翻譯」中，以最靠近缺失位置的短錨點字詞（通常為缺失應落點的前一詞；必要時後一詞）加上 `<spanN>...</spanN>`。
	* 不新增虛構文字、不使用空標籤；一處缺失對應一個非重疊 `<spanN>`。
* 當「機器翻譯」完全正確時，`errors_in_mt` 應回傳**未插入任何 `<span>`** 的原始 mt 字串，且 `errors` 為空陣列。
* 特別注意俗語、成語和固定用法。機器翻譯必須使用目標語言中對等且自然的慣用語，而非僅僅直譯其字面意義。
* **Alignment Hints 優先級**：當 Alignment Hints 中提供了與原文相似的句子或詞彙時，該提示應被優先參考。任何與提示在「形式」或「用詞」上有顯著差異的機器翻譯，即使語意相近，也應被標記為錯誤（通常是 Style/Awkward 或 Terminology/Inappropriate），尤其需要特別注意俗語、成語和台語固定用法。


## RULES（務必遵守）

* `errors_in_mt` 中每一個 `<spanN>...</spanN>`：
  * **必須**有且僅有一個對應的 `</spanN>`；
  * **必須**在 `errors` 陣列中有**一筆** `error_id = N` 的錯誤記錄。
* **不得巢狀**與**不得重疊**任兩個標籤。必要時改為**相鄰不重疊**的較小片段。
* `explanation` 應在最後給出一個整體的解釋即可，不需要在每個錯誤標註後都給出。
	* 當提及被標記的錯誤時，必須引用其「文字內容」不得直接寫出 `<spanN>` 標籤本身。 
* 在 `score` 整體評分中，儘管句子中可能出現許多錯誤片段，但若是**仍能看出翻譯大致意思**，就不應該給予過低的分數，應傾向**給予接近60分**。
* 標籤語法**只能**為 `<spanN>` 與 `</spanN>`（注意是**斜線 `/`**，不可用反斜線 `\`；`N` 必須是阿拉伯數字，且與 `errors.error_id` 一致）。
* `severity` 與 `error_type` **一律只用英文**，不得附上括號或中文註解（例如 `Omission`，不可寫成 `Omission（漏譯）`）。
* 僅回傳**單一 JSON 物件**；不得出現多餘文字。
* 請勿輸出幻覺內容，所有標註皆須有依據。

## 輸出格式（STRICT）

只回傳以下結構的單一 JSON 物件：

{{
  "errors_in_mt": "<translation_string_with_<spanN>...</spanN>_tags_or_original_if_no_errors>",
  "errors": [
    {{
      "error_id": <int>,               // 與 <spanN> 的 N 對應
      "error_text": "<spanN>", // 字面量標籤名，例如 "<span1>"
      "severity": "Major" | "Minor",
      "error_type": "<error_catogory/subcategory>"
    }}
    // 0 個或多個
  ],
  "explanation": "<說明每個錯誤片段&整體建議翻譯>",
  "score": <int>
}}
"""

USER_PROMPT = """請根據以下「原文」與「機器翻譯」進行錯誤偵測與文字內嵌標記，並輸出符合規範的 JSON。
特別注意 **Alignment Hints 優先級**：當 Alignment Hints 中提供了與原文相似的句子或詞彙時，該提示應被優先參考。任何與提示在「形式」或「用詞」上有顯著差異的機器翻譯，即使語意相近，也應被標記為錯誤（通常是 Style/Awkward 或 Terminology/Inappropriate），尤其需要特別注意俗語、成語和台語固定用法。

原文（src）：{src}
機器翻譯（mt）：{mt}
"""



# ===== Retrieval helper: BGE-M3 (dense embedding) =====
from FlagEmbedding import BGEM3FlagModel

class BGEM3Retriever:
    def __init__(self, texts: List[str], model_name: str = 'BAAI/bge-m3', use_fp16: bool = True):
        self.texts = texts
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        # Batch-encode and L2-normalize (so cosine == dot product)
        out = self.model.encode(
            self.texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        emb = out['dense_vecs'].astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        self.emb = emb / norms

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_out = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        q = q_out['dense_vecs'][0].astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self.emb @ q  # cosine
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

# ===== Retrieval helper: multilingual-e5-base (dense embedding) =====
from sentence_transformers import SentenceTransformer
import torch

class E5Retriever:
    """
    Drop-in replacement for BGEM3Retriever.
    - Encodes the corpus once with a 'passage: ' prefix and L2-normalizes it.
    - Encodes each query with a 'query: ' prefix and L2-normalizes it.
    - Ranks by cosine similarity (dot product).
    """
    def __init__(self, texts: List[str], model_name: str = "intfloat/multilingual-e5-base",
                 batch_size: int = 128, device: Optional[str] = None):
        self.texts = texts
        self.model = SentenceTransformer(model_name, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # e5 requires a 'passage: ' prefix on the corpus
        passages = [("passage: " + t) for t in self.texts]
        emb = self.model.encode(passages,
                                batch_size=batch_size,
                                normalize_embeddings=True,   # L2-normalize directly
                                convert_to_numpy=True,
                                show_progress_bar=False)
        self.emb = emb.astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([f"query: {query}"],
                                  normalize_embeddings=True,
                                  convert_to_numpy=True,
                                  show_progress_bar=False)[0].astype(np.float32)
        scores = self.emb @ q_emb  # cosine (already normalized)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


def robust_json_loads(text: str) -> Dict[str, Any]:
    """General robust JSON loader."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

        # First use a regex to extract the JSON-looking part
        m = JSON_RE.search(text)
        candidate = None
        if m:
            candidate = m.group(0)
        else:
            # remove "```"
            text2 = text.strip().strip("`").strip()
            m2 = JSON_RE.search(text2)
            if m2:
                candidate = m2.group(0)

        # If a candidate was found, first try a plain json.loads
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # If it is still broken, repair with json_repair and load again
                repaired = repair_json(candidate)
                return json.loads(repaired)

        # Reaching here means the regex found no {}; repair the whole text
        repaired = repair_json(text)
        return json.loads(repaired)

# ===== Utility: parse <spanN>...</spanN> and compute indices =====
START_TAG_RE = re.compile(r"<span(\d+)>")
# END_TAG_RE   = re.compile(r"</span(\d+)>")
END_TAG_RE = re.compile(r"</span(\d*)>|<span(\d+)>")

class TagParseError(Exception):
    pass

def parse_tagged_translation(annotated: str) -> Tuple[str, Dict[int, Tuple[int, int]]]:
    """
    Parse a string containing <spanN>...</spanN> tags into:
      - clean_text: the text with all tags removed
      - spans_by_id: dict { N: (start, end) } indexed into clean_text
    Rules:
      - Multiple non-overlapping tags are allowed; nesting/crossing is not
        (handled defensively via a stack / nearest match if it occurs).
      - Any unpaired or malformed tag raises TagParseError.
    """

    i = 0
    L = len(annotated)
    out_chars: List[str] = []
    spans_by_id: Dict[int, Tuple[int, int]] = {}
    stack: List[Tuple[int, int]] = []  # (id, start_pos_in_out)

    while i < L:
        # Try to match opening/closing tags
        m_start = START_TAG_RE.match(annotated, i)
        if m_start:
            span_id = int(m_start.group(1))
            # Do not allow reopening an already-open id -> treat as nested/overlapping
            if any(sid == span_id for sid, _ in stack):
                # Raise immediately to avoid corrupting indices
                raise TagParseError(f"Nested or repeated opening for span{span_id} at pos {i}")
            stack.append((span_id, len(out_chars)))
            i = m_start.end()
            continue

        m_end = END_TAG_RE.match(annotated, i)
        if m_end:
            span_id, start_pos = stack.pop()
            spans_by_id[span_id] = (start_pos, len(out_chars))
            i = m_end.end()
            continue

        # Normal character -> output
        out_chars.append(annotated[i])
        i += 1

    if stack:
        # Unclosed tags remain -> treat as an error
        unclosed = ",".join(str(sid) for sid, _ in stack)
        raise TagParseError(f"Unclosed tags for span IDs: {unclosed}")

    clean_text = "".join(out_chars)
    print(f"clean_text: {clean_text}, spans_by_id: {spans_by_id}")
    return clean_text, spans_by_id


# ===== Post-processing: tags -> indices =====
def postprocess_to_spans(mt: str, model_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the model output object into the final format:
      {
        "errors": [
          {"error_text": "...", "start": int, "end": int, "severity": "...", "error_type": "..."}
        ]
      }
    """
    annotated = model_obj.get("errors_in_mt", mt)
    listed = model_obj.get("errors", [])

    try:
        clean_text, spans_by_id = parse_tagged_translation(annotated)
    except TagParseError:
        # If tags are broken, fall back to treating the input as untagged and return empty
        clean_text, spans_by_id = mt, {}

    # Safety: if the cleaned text != original mt, the model altered untagged content; keep the original mt and drop indices
    if len(clean_text.strip()) != len(mt.strip()):
        spans_by_id = {}
        clean_text = mt

    # Build the results
    final_errors: List[Dict[str, Any]] = []

    for item in listed:
        # Require error_id and severity/error_type
        span_tag = item.get("error_text", "")
        m = re.fullmatch(r"<span(\d+)>", span_tag or "")
        if not m:
            continue
        eid = int(m.group(1))
        sev = item.get("severity", "Major")
        ety = item.get("error_type", "Mistranslation")

        if eid in spans_by_id:
            s, e = spans_by_id[eid]
            # Boundary check
            if 0 <= s < e <= len(mt):
                final_errors.append({
                    "error_text": mt[s:e],
                    "start": s,
                    "end": e,
                    "severity": sev,
                    "error_type": ety
                })

    # Sort by start position and deduplicate (same range and type = duplicate)
    seen = set()
    deduped = []
    for err in sorted(final_errors, key=lambda x: (x["start"], x["end"], x["error_type"], x["severity"])):
        key = (err["start"], err["end"], err["error_type"], err["severity"])
        if key not in seen:
            seen.add(key)
            deduped.append(err)

    return {"errors": deduped}

# ------------------------------------------------------------------------------
# Extract jieba tokens (length >= 2), dedup while preserving order
# ------------------------------------------------------------------------------
def extract_tokens_zh(text: str, min_len: int = 2, max_tokens: int = 30) -> List[str]:
    toks = jieba.lcut(text, cut_all=False)
    seen, kept = set(), []
    for t in toks:
        t = t.strip()
        # Keep only CJK or alphanumeric tokens with length >= min_len
        if len(t) >= min_len and t not in seen:
            kept.append(t)
            seen.add(t)
        if len(kept) >= max_tokens:
            break
    return kept

# ------------------------------------------------------------------------------
# Assemble the retrieval results into the Alignment Hints text block
# ------------------------------------------------------------------------------
def build_alignment_hints_text(
    sent_items: List[Dict[str, str]],
    word_items: List[Dict[str, str]],
) -> str:
    """
    sent_items: [{zh, han, tl}], at most 3-5 entries
    word_items: [{zh, han, tl}], a few entries
    """
    # Sentence-level block
    sent_lines = []
    for i, it in enumerate(sent_items, 1):
        zh = it.get("zh", "")
        han = it.get("han", "")
        tl  = it.get("tl", "")
        line = (
            f" Mandarin: {zh}\n"
            f" Taiwanese(HAN): {han}\n"
            f" Taiwanese(TL, optional): {tl}\n"
        )
        sent_lines.append(line)

    # Word-level block
    word_lines = []
    for it in word_items:
        zh = it.get("zh", "")
        han = it.get("han", "")
        tl  = it.get("tl", "")
        if zh:  # keyed by zh
            if tl:
                word_lines.append(f"{zh} = tai(HAN: {han}; TL: {tl}, optional);")
            else:
                word_lines.append(f"{zh} = tai(HAN: {han}; TL: , optional);")

    # Final text
    text = (
        "## Alignment Hints\n"
        "> 僅供推理參考，請勿在此區塊內插入任何 <spanN> 標籤；本區塊不改變前述所有 RULES\n\n"
        "### 使用規則：\n"
        "* Sentence-level alignment：提供少量「中文 <-> 台文」對齊句，輔助語義對照；若與原文衝突，以原文為準。\n"
        "* Word-level alignment：提供中文、台文（HAN）、台羅（TL）詞彙對照，協助檢出每個錯誤片段。\n\n"
        "### Sentence-level alignment（few-shot, zh<->tai）\n"
        "```\n"
        + "\n".join(sent_lines) +
        "```\n\n"
        "### Word-level alignment（lexical hints, zh->tai）\n"
        "```\n"
        + "\n".join(word_lines) +
        "\n```"
    )
    return text

# ------------------------------------------------------------------------------
# Produce both retrieval results from the source sentence
# ------------------------------------------------------------------------------
def make_sentence_level_hints(
    src_zh: str,
    df_sent: pd.DataFrame,
    sent_retriever: E5Retriever,
    top_k: int = 3,
    min_score: float = 0.4,
) -> List[Dict[str, str]]:
    """
    Retrieve the most similar Chinese sentences from sentences.json and
    return their {zh, han, tl}.
    - Fetch a few extra first, then filter by min_score.
    - Keep at most top_k sentences.
    """
    # Fetch a few extra so enough remain after thresholding
    raw = sent_retriever.search(src_zh, top_k=top_k * 3)

    # Keep only items with score >= min_score
    filtered = [(idx, score) for idx, score in raw if score >= min_score]

    # Trim to at most top_k
    filtered = filtered[:top_k]

    ans: List[Dict[str, str]] = []
    for idx, score in filtered:
        row = df_sent.iloc[idx]
        ans.append({
            "zh": str(row.get("中文", "")),
            "han": str(row.get("台文", "")),
            "tl": str(row.get("台羅", "")),
            "_score": float(score),  # Optionally include scores for debugging
        })
    # print(f"Sentence-level hints: {ans}")
    return ans

def make_word_level_hints(
    src_zh: str,
    df_word: pd.DataFrame,
    word_retriever: E5Retriever,
    per_token_top: int = 1,
    min_len: int = 2,
    max_total: int = 40,
    min_score: float = 0.4,
) -> List[Dict[str, str]]:
    """
    - Tokenize with jieba (length >= min_len).
    - For each token, use the retriever to find the per_token_top most
      similar Chinese words.
    - Keep only those with score >= min_score.
    - Keep at most max_total in total.
    """
    toks = extract_tokens_zh(src_zh, min_len=min_len, max_tokens=max_total * 2)
    results: List[Dict[str, str]] = []

    for tok in toks:
        if len(results) >= max_total:
            break

        # Query top-k for each token
        top = word_retriever.search(tok, top_k=per_token_top)

        for idx, score in top:
            if score < min_score:
                continue  # Skip if the score is too low

            row = df_word.iloc[idx]
            results.append({
                "zh": str(row.get("中文", "")),
                "han": str(row.get("台文", "")),
                "tl": str(row.get("台羅", "")),
                "_score": float(score),  # Enable for debugging
            })

            if len(results) >= max_total:
                break
    print(f"Word-level hints: {results}")
    return results

# ------------------------------------------------------------------------------
# Build the Alignment Hints string (optionally added to the prompt)
# ------------------------------------------------------------------------------
def build_alignment_hints_block(
    src_zh: str,
    df_sent: pd.DataFrame,
    df_word: pd.DataFrame,
    sent_retriever: E5Retriever,
    word_retriever: E5Retriever,
    sent_top_k: int = 3,
    per_token_top: int = 1,
    sent_min_score: float = 0.4,
    word_min_score: float = 0.4,
    word_max_total: int = 40,
) -> str:
    sent_items = make_sentence_level_hints(
        src_zh,
        df_sent,
        sent_retriever,
        top_k=sent_top_k,
        min_score=sent_min_score,
    )
    word_items = make_word_level_hints(
        src_zh,
        df_word,
        word_retriever,
        per_token_top=per_token_top,
        min_len=2,
        max_total=word_max_total,
        min_score=word_min_score,
    )
    return build_alignment_hints_text(sent_items, word_items)

# ------------------------------------------------------------------------------
# Randomly sample Alignment Hints (no retriever, no similarity threshold)
# ------------------------------------------------------------------------------
def make_random_sentence_level_hints(
    df_sent: pd.DataFrame,
    top_k: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Randomly sample top_k sentences from df_sent and return [{zh, han, tl}].
    - No retriever, no similarity threshold.
    - seed: optional random seed; None means a different draw each run.
    """
    n = len(df_sent)
    k = min(top_k, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    ans: List[Dict[str, str]] = []
    for idx in indices:
        row = df_sent.iloc[int(idx)]
        ans.append({
            "zh":  str(row.get("中文", "")),
            "han": str(row.get("台文", "")),
            "tl":  str(row.get("台羅", "")),
        })
    return ans


def make_random_word_level_hints(
    src_zh: str,
    df_word: pd.DataFrame,
    per_token_top: int = 1,
    min_len: int = 2,
    max_total: int = 40,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Same as make_word_level_hints, but after tokenizing src_zh with jieba it
    randomly samples per_token_top words for each token (instead of the most
    semantically similar ones), keeping at most max_total in total.
    - No similarity threshold, no retriever.
    - seed: optional random seed; None means a different draw each run.
    """
    toks = extract_tokens_zh(src_zh, min_len=min_len, max_tokens=max_total * 2)
    results: List[Dict[str, str]] = []
    n = len(df_word)
    rng = np.random.default_rng(seed)

    for tok in toks:
        if len(results) >= max_total:
            break

        # Randomly sample per_token_top words for each token
        k = min(per_token_top, n)
        indices = rng.choice(n, size=k, replace=False)

        for idx in indices:
            row = df_word.iloc[int(idx)]
            results.append({
                "zh":  str(row.get("中文", "")),
                "han": str(row.get("台文", "")),
                "tl":  str(row.get("台羅", "")),
            })
            if len(results) >= max_total:
                break

    return results


# ===== Model call =====
def call_model(src: str, mt: str, additional_context: Optional[str] = None) -> Dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_content = USER_PROMPT.format(src=src, mt=mt)
    if additional_context:
        user_content = f"{user_content}\n\n{additional_context}"
    messages.append({"role": "user", "content": user_content})


    # for gpt-5
    response = client.responses.create(
        model=MODEL_NAME,
        input=messages, 
        reasoning={"effort": "medium"},   # ("low" | "medium" | "high")
        text={"verbosity": "low"}      # ("low" | "medium" | "high")
    )
    message_item = next(o for o in response.output if o.type == "message")
    output_text = message_item.content[0].text.strip()
    print(f"Model output: {output_text}")

    try:
        obj = robust_json_loads(output_text)
    except Exception as e:
        print(f"⚠️Error parsing JSON from model output: {e}, errors_in_mt will be set to original mt.")
        obj = {"errors_in_mt": mt, "errors": []}

    if "errors_in_mt" not in obj or not isinstance(obj["errors_in_mt"], str):
        obj["errors_in_mt"] = mt
    if "errors" not in obj or not isinstance(obj["errors"], list):
        obj["errors"] = []

    return obj



# ===== Main pipeline =====
def process_one_record(d: pd.Series,
                       alignment_hints: Optional[str] = None) -> Dict[str, Any]:
    print(f"Processing ID: {d['id']}")
    model_obj = call_model(d["source"], d["target"], additional_context=alignment_hints)
    error_spans = postprocess_to_spans(d["target"], model_obj) # error_spans
    explanation = model_obj.get("explanation", "")
    score = model_obj.get("score", 0)
    print(f"error_spans: {error_spans}")
    return error_spans, explanation, score

def main():
    input_path = "moedict_select-taigi.json"
    output_path = f"./pred_output/{MODEL_NAME}_random_sent3_word1.jsonl"

    data = pd.read_json(input_path, encoding='utf-8').T

    # ===============================================================================
    # Load the sentences / words datasets (columns "中文" / "台文" / "台羅")
    dataset_sentence = "../data/sentences.json"
    with open(dataset_sentence, 'r', encoding='utf-8') as f:
        df_sentence = pd.read_json(f)

    dataset_word = "../data/words.json"
    with open(dataset_word, 'r', encoding='utf-8') as f:
        df_word = pd.read_json(f)

    #                             model_name="intfloat/multilingual-e5-base")
    #                             model_name="intfloat/multilingual-e5-base")

    # ==== Toggle: whether to feed Alignment Hints to the LLM ====
    USE_ALIGNMENT_HINTS = True
    USE_SENT_RETRIEVER = False   
    USE_WORD_RETRIEVER = False  
    USE_RANDOM_SENT_RETRIEVER = True  # Randomly sample sentences (no embedding similarity)
    USE_RANDOM_WORD_RETRIEVER = True  # Randomly sample words (no embedding similarity)

    SENT_TOP_K = 3        # Number of sentence-alignment few-shots (also used for random sent)
    SENT_MIN_SCORE = 0.8     # Sentence similarity threshold (cosine; drop below this)

    WORD_PER_TOKEN_TOP = 1  # Max matched words per Chinese token
    WORD_MIN_SCORE = 0.85   # Word similarity threshold (cosine)
    WORD_MAX_TOTAL = 100   # Max total word-level alignments (also used for random word)
    # ===============================================================================

    # ==== Mutual-exclusion check ====
    assert not (USE_SENT_RETRIEVER and USE_RANDOM_SENT_RETRIEVER), \
        "USE_SENT_RETRIEVER and USE_RANDOM_SENT_RETRIEVER cannot be enabled at the same time"
    assert not (USE_WORD_RETRIEVER and USE_RANDOM_WORD_RETRIEVER), \
        "USE_WORD_RETRIEVER and USE_RANDOM_WORD_RETRIEVER cannot be enabled at the same time"

    # if os.path.exists(output_path):
    #     import datetime
    #     output_path = output_path + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))

    for _, d in tqdm(data.iterrows(), total=len(data)):
        # Build Alignment Hints (from the source Chinese)
        alignment_hints = None
        if USE_ALIGNMENT_HINTS:
            sent_items = []
            word_items=[]

            if USE_SENT_RETRIEVER:
                sent_items = make_sentence_level_hints(
                    src_zh=d["source"],
                    df_sent=df_sentence,
                    sent_retriever=sent_retriever,
                    top_k=SENT_TOP_K,
                    min_score=SENT_MIN_SCORE,
                )
            if USE_WORD_RETRIEVER:
                word_items = make_word_level_hints(
                    src_zh=d["source"],
                    df_word=df_word,
                    word_retriever=word_retriever,
                    per_token_top=WORD_PER_TOKEN_TOP,
                    min_len=2,
                    max_total=WORD_MAX_TOTAL,
                    min_score=WORD_MIN_SCORE,
                )
            if USE_RANDOM_SENT_RETRIEVER:
                random_sent_items = make_random_sentence_level_hints(
                    df_sent=df_sentence,
                    top_k=SENT_TOP_K,
                )
                sent_items = sent_items + random_sent_items
            if USE_RANDOM_WORD_RETRIEVER:
                random_word_items = make_random_word_level_hints(
                    src_zh=d["source"],
                    df_word=df_word,
                    per_token_top=WORD_PER_TOKEN_TOP,
                    min_len=2,
                    max_total=WORD_MAX_TOTAL,
                )
                word_items = word_items + random_word_items


            alignment_hints = build_alignment_hints_text(sent_items, word_items)

        error_spans, explanation, score = process_one_record(d, alignment_hints=alignment_hints)

        ordered = OrderedDict()
        ordered["id"] = d.get("id")
        ordered["src"] = d.get("source")
        ordered["mt"] = d.get("target")
        ordered["ref"] = d.get("reference")
        ordered.update(error_spans)
        ordered["explanation"] = explanation
        ordered["score100"] = score
        ordered["score7"] = round(score*(6/100), 2)

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ordered, ensure_ascii=False) + '\n')

    print("=========== Done! ============")

if __name__ == "__main__":
    main()
