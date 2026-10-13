"""CiteSure: the one class both the UI and the API call (Phases 4-7)."""
from .config import settings
from .ingest import ingest_pdf
from .llm import complete
from .prompts import (COMPARE_SYSTEM, FACETS, NOT_FOUND, SYSTEM, format_context,
                      parse_citations)
from .store import PaperStore

REWRITE_SYSTEM = ("Rewrite the user's last question as a standalone search query, resolving "
                  "pronouns using the conversation. Output only the query, nothing else.")

SECTION_TERMS = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "methodology": ["methodology", "methods", "proposed method"],
    "results": ["results", "experiments", "findings"],
    "limitations": ["limitations", "threats to validity"],
    "conclusion": ["conclusion", "concluding remarks"],
}


def detect_section(question):
    q = question.lower().strip().rstrip("?.!")

    patterns = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "background section"],
        "methodology": ["methodology", "methods used", "research design"],
        "results": ["results", "findings", "experimental results"],
        "limitations": ["limitations", "weaknesses", "threats to validity"],
        "conclusion": ["conclusion", "concluding remarks"],
    }

    for section, phrases in patterns.items():
        if any(p in q for p in phrases):
            return section

    return None

def standalone(history, question: str) -> str:
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    return complete(REWRITE_SYSTEM, f"{convo}\nuser: {question}", max_tokens=80).strip()


class CiteSure:
    def __init__(self):
        self.store = PaperStore()

    # ---- indexing -------------------------------------------------------------
    def add_pdf(self, path: str) -> dict:
        chunks = ingest_pdf(path)
        if not chunks:
            raise ValueError("No extractable text found. Is this a scanned PDF?")
        pid = chunks[0].paper_id
        if not self.store.has_paper(pid):
            self.store.add_chunks(chunks)
        return {"paper_id": pid, "title": chunks[0].title, "chunks": len(chunks)}

    # ---- question answering ---------------------------------------------------
    def retrieve(self, query: str, paper_ids=None):
        k = 20 if settings.use_rerank else settings.top_k
        if settings.use_hybrid:
            from .hybrid import hybrid_search
            hits = hybrid_search(self.store, query, paper_ids, k=k)   # fused hits have score 0
        else:
            hits = [h for h in self.store.search(query, k=k, paper_ids=paper_ids)
                    if h["score"] >= settings.min_score]
        if hits and settings.use_rerank:
            from .rerank import rerank
            hits = rerank(query, hits, top_n=settings.top_k)
        return hits

    def ask(self, question: str, paper_ids=None, history=None) -> dict:
        query = standalone(history, question)
        hits = self.retrieve(query, paper_ids)
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "invalid_citations": [], "query": query}
        user = f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}"
        answer = complete(SYSTEM, user)
        _, invalid = parse_citations(answer, hits)
        return {"answer": answer, "sources": hits, "invalid_citations": invalid, "query": query}

    # ---- summarization --------------------------------------------------------
    def summarize(self, paper_id: str) -> dict:
        sections = []
        for name, question in FACETS.items():
            hits = self.store.search(question, k=4, paper_ids=[paper_id])
            user = (f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}\n"
                    "Answer in 2-4 sentences.")
            sections.append((name, complete(SYSTEM, user, max_tokens=500), hits))
        digest = "\n\n".join(f"{n}: {t}" for n, t, _ in sections)
        tldr = complete("Write a 3-sentence TL;DR of a research paper from these notes. "
                        "Do not include citation numbers.", digest, max_tokens=300)
        return {"tldr": tldr, "sections": sections}

    # ---- comparison -----------------------------------------------------------
    def compare(self, paper_ids, aspects) -> dict:
        found = {}
        for aspect in aspects:
            for pid in paper_ids:               # per-paper retrieval = fair comparison
                for h in self.store.search(aspect, k=3, paper_ids=[pid]):
                    found[h["id"]] = h          # dict de-duplicates overlapping hits
        hits = list(found.values())
        titles = sorted({h["meta"]["title"] for h in hits})
        user = (f"CONTEXT:\n{format_context(hits)}\n\nPAPERS: {'; '.join(titles)}\n"
                f"ASPECTS: {', '.join(aspects)}\nProduce the comparison.")
        answer = complete(COMPARE_SYSTEM, user, max_tokens=1800)
        _, invalid = parse_citations(answer, hits)
        return {"answer": answer, "sources": hits, "invalid_citations": invalid}
