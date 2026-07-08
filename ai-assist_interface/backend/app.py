from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, json
import tempfile, time, shutil

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
   allow_origins=[
    "http://203.145.216.166:50073",    
    "http://localhost:5000"            
  ],
  allow_credentials=True,
  allow_methods=["*"],          
  allow_headers=["*"],         
)

@app.get("/") 
def root():
    return {
        "status": "ok",
        "message": "Annotation backend is running.",
        "endpoints": ["/annotation/save (POST)", "/annotation/load (GET)", "/docs"]
    }

@app.get("/healthz")  
def healthz():
    return JSONResponse({"status": "ok"})

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
ANNOTATION_DIR = (BASE_DIR / "annotations").as_posix()
os.makedirs(ANNOTATION_DIR, exist_ok=True)

@app.post("/annotation/save")
async def save_annotation(req: Request):
    data = await req.json()
    base_name = os.path.splitext(data["filename"])[0]
    filename = base_name
    rater = data["rater"]
    record = data["record"]
    key = str(record["id"])

    path = os.path.join(ANNOTATION_DIR, f"{filename}_{rater}_annotation.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_records = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_records = {}
    else:
        all_records = {}


    all_records[key] = record

    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            bdir = os.path.join(ANNOTATION_DIR, ".backup", os.path.basename(path))
            os.makedirs(bdir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(path, os.path.join(bdir, f"{os.path.basename(path)}.{ts}.json"))
    except Exception:

        pass

    fd, tmppath = tempfile.mkstemp(prefix=os.path.basename(path)+".", dir=ANNOTATION_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
            json.dump(all_records, tmpf, ensure_ascii=False, indent=2)
            tmpf.flush()
            os.fsync(tmpf.fileno())
        os.replace(tmppath, path)  
    finally:
        if os.path.exists(tmppath):
            try: os.remove(tmppath)
            except: pass

    return {"status": "ok"}

@app.get("/annotation/load")
async def load_annotation(filename: str, rater: str, id: int):
    filename = os.path.splitext(filename)[0]
    path = os.path.join(ANNOTATION_DIR, f"{filename}_{rater}_annotation.json")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {"record": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            all_records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"record": None}
    return {"record": all_records.get(str(id))}
