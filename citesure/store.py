"""Embeddings + ChromaDB wrapper (Phase 3)."""
import chromadb
from sentence_transformers import SentenceTransformer

from .config import settings

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_model = None


def embedder() -> SentenceTransformer:
    global _model
    if _model is None:                          # lazy-load: first call downloads ~130 MB
        _model = SentenceTransformer(settings.embed_model)
    return _model


def embed_docs(texts):
    return embedder().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(q):
    return embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()


def _where(paper_ids):
    return {"paper_id": {"$in": paper_ids}} if paper_ids else None


class PaperStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.col = self.client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})

    def has_paper(self, paper_id: str) -> bool:
        return len(self.col.get(where={"paper_id": paper_id}, limit=1)["ids"]) > 0

    def add_chunks(self, chunks):
        self.col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed_docs([c.text for c in chunks]),
            metadatas=[{"paper_id": c.paper_id, "title": c.title,
                        "section": c.section, "page": c.page} for c in chunks],
        )

    def search(self, query: str, k: int = 6, paper_ids=None):
        res = self.col.query(query_embeddings=[embed_query(query)], n_results=k,
                             where=_where(paper_ids))
        return [{"id": res["ids"][0][i], "text": res["documents"][0][i],
                 "meta": res["metadatas"][0][i], "score": 1 - res["distances"][0][i]}
                for i in range(len(res["ids"][0]))]

    def get_chunks(self, paper_ids=None):
        r = self.col.get(where=_where(paper_ids), include=["documents", "metadatas"])
        return [{"id": i, "text": t, "meta": m, "score": 0.0}
                for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])]

    def list_papers(self):
        seen = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            p = seen.setdefault(m["paper_id"], {"paper_id": m["paper_id"], "title": m["title"], "chunks": 0})
            p["chunks"] += 1
        return list(seen.values())

    def delete_paper(self, paper_id: str):
        self.col.delete(where={"paper_id": paper_id})
