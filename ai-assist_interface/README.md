# AI-Assisted Human Annotation

A lightweight web tool for **post-editing machine-translation error-span annotations**.
For each machine translation, the interface displays the AI-proposed error spans produced
by the Cross-lingual Semantic Scaffolding pipeline, and lets expert annotators verify and
correct them. This shifts the annotator's task from *searching* for errors to *verifying*
them, substantially reducing annotation effort.

## Screenshot
<img width="1912" height="893" alt="image" src="https://github.com/user-attachments/assets/f66279c9-9368-4473-8f22-bc14a799eea5" />

## What it does

- Loads sentence pairs (Mandarin source, Taiwanese Hokkien machine translation) with
  AI pre-annotated error spans overlaid on the translation.
- The annotator reviews each span and edits it using the operations below.
- Every action is logged, and the corrected spans, severity labels, segment-level score,
  and explanation are saved as a JSON file under `backend/annotations/`.

## Annotation operations

Each operation acts on an AI-proposed error span shown on the machine translation:

- **Adopt (No-edit).** Accept an AI-proposed span unchanged.
- **Add.** Mark a new error span that the model did not propose.
- **Remove.** Delete an AI-proposed span (e.g., a false positive).
- **Resize.** Adjust the boundaries of an existing span — extending or shrinking its
  start/end offsets — so that it covers exactly the erroneous tokens, without relocating
  the span.
- **Move.** Shift an existing span to a different position in the sentence, correcting its
  location while keeping the span itself.
- **Reset.** Revert a span to its original AI-proposed state, undoing any manual edits made
  to it. (Spans that were reset therefore ultimately correspond to AI proposals adopted in
  their original form.)
- **Severity.** Change a span's severity label (`Minor` / `Major` / `No-error`).

## How to start

### Backend (port 8000)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend (port 5000)
```bash
npm run dev -- --hostname 0.0.0.0 --port 5000
```

## File structure

```
project-root/
│
├── frontend/           # Next.js / React app (run on port 5000 with `npm run dev`)
│    └── ...
│
├── backend/            # Backend API
│   ├── app.py          # FastAPI main application
│   └── annotations/    # Stored annotation JSON files
│
└── README.md
```
