# AI-Assisted Human Annotation

## Screenshot
<img width="1912" height="893" alt="image" src="https://github.com/user-attachments/assets/f66279c9-9368-4473-8f22-bc14a799eea5" />


## How to start
### [backend] use port 8000
`uvicorn app:app --host 0.0.0.0 --port 8000`

### [frontend] use port 5000
`npm run dev -- --hostname 0.0.0.0 --port 5000`


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
