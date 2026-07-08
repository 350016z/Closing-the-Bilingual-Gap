# Retrieval Corpora

This directory contains the retrieval corpora used to build the **Alignment Hints**
(sentence-level and word-level Chinese↔Taiwanese references).

> **Note on data availability.** Due to licensing and confidentiality constraints,
> the full corpora **cannot** be redistributed here. Each file below is provided
> only as a small **sample (~20 entries)** so that the data format is clear and the
> code can be run end-to-end. The complete data can be reconstructed from the public
> sources listed below.

## Files

| File | Full size | Sample here | Description |
|------|-----------|-------------|-------------|
| `sentences.json` | 14,095 entries | 20 | Sentence-level Chinese↔Taiwanese pairs |
| `words.json` | 38,260 entries | 20 | Word-level Chinese↔Taiwanese lexical entries |

### Schema

`sentences.json` — list of objects:

```json
{ "台文": "...", "台羅": "...", "中文": "...", "id": 0 }
```

`words.json` — list of objects (`詞性` = part of speech, `解說` = gloss/definition;
both may be `null`):

```json
{ "台文": "...", "台羅": "...", "中文": "...", "詞性": null, "解說": null, "id": 0 }
```

Field meaning: `台文` = Taiwanese (Han characters), `台羅` = Tâi-lô romanization,
`中文` = Mandarin correspondence.

## Data sources

### Sentence-level retrieval (14,243 entries before deduplication)

- [教育部臺灣台語常用詞辭典](https://sutian.moe.edu.tw/zh-hant/) — 11,582 entries
- [TAT Corpus](https://sites.google.com/speech.ntut.edu.tw/fsw/home/tat-corpus) — 2,661 entries

### Word-level retrieval (38,260 entries)

- [教育部臺灣台語常用詞辭典](https://sutian.moe.edu.tw/zh-hant/) — 34,380 entries
  (with semi-automatic lexical expansion via Gemini 2.5 Pro; see note below)
- [臺灣台語推薦用字700字詞](https://language.moe.gov.tw/result.aspx?classify_sn=23&subclassify_sn=439) — 1,203 entries
- [學科術語臺灣台語/臺灣客語對譯查詢](https://stti.moe.edu.tw/file-download/newsData/?lang=sutgi) — 2,677 entries

### Note on the Gemini 2.5 Pro semi-automatic expansion

Some entries in [教育部臺灣台語常用詞辭典](https://sutian.moe.edu.tw/zh-hant/) do not
provide a direct Mandarin word correspondence (`中文`) for the Taiwanese term.
For these entries we did **not** translate the Taiwanese form (`台文`) into Mandarin
directly with Gemini 2.5 Pro. Instead, we fed Gemini 2.5 Pro the Mandarin
**definition / explanatory gloss** (`中文解釋`) that the dictionary already provides
for the entry, and asked it to convert that definition into concise Mandarin
**word(s) / phrase(s)** (`中文字詞`). This yields a usable `中文` field while staying
faithful to the dictionary's own glosses.
