import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from citesure.config import settings
from citesure.service import CiteSure

app = FastAPI(title="CiteSure AI API", version="1.0")
pp = CiteSure()
MAX_MB = 25


class AskIn(BaseModel):
    question: str = Field(min_length=3)
    paper_ids: list[str] | None = None
    history: list[tuple[str, str]] = []


class CompareIn(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    aspects: list[str] = ["problem", "method", "datasets", "results", "limitations"]


@app.post("/papers")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    os.makedirs(settings.upload_dir, exist_ok=True)
    path = os.path.join(settings.upload_dir, os.path.basename(file.filename))
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    if os.path.getsize(path) > MAX_MB * 1024 * 1024:
        os.remove(path)
        raise HTTPException(413, f"File larger than {MAX_MB} MB.")
    try:
        return pp.add_pdf(path)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/papers")
def list_papers():
    return pp.store.list_papers()


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    pp.store.delete_paper(paper_id)
    return {"deleted": paper_id}


@app.post("/ask")
def ask(body: AskIn):
    return pp.ask(body.question, body.paper_ids, body.history)


@app.get("/papers/{paper_id}/summary")
def summary(paper_id: str):
    return pp.summarize(paper_id)


@app.post("/compare")
def compare(body: CompareIn):
    return pp.compare(body.paper_ids, body.aspects)
