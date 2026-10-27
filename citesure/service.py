"""CiteSure: the one class both the UI and the API call (Phases 4-7)."""
import re

from .config import settings
from .ingest import ingest_pdf
from .llm import complete
from .prompts import (COMPARE_SYSTEM, FACETS, NOT_FOUND, SYSTEM, format_context,
                      parse_citations)
from .store import PaperStore

REWRITE_SYSTEM = ("Rewrite the user's last question as a standalone search query, resolving "
                  "pronouns using the conversation. Output only the query, nothing else.")

SECTION_PATTERNS = {
    "abstract": r"\babstract\b|\basbstract\b|\babstarct\b",
    "introduction": r"\bintroduction\b|\bbackground\b",
    "methodology": r"\bmethodology\b|\bmethods?\b|\bapproach\b",
    "results": r"\bresults?\b|\bfindings\b|\bexperiments?\b",
    "limitations": r"\blimitations?\b|\bweaknesses\b",
    "conclusion": r"\bconclusions?\b|\bconcluding remarks\b",
}


def detect_section(question: str):
    """Return a section name only when the user clearly asks for that section."""
    q = question.lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, q) and (section == "abstract" or "section" in q):
            return section
    return None


def standalone(history, question: str) -> str:
    if not history:
        return question
    convo = "\n".join(f"{role}: {text}" for role, text in history[-6:])
    rewritten = complete(REWRITE_SYSTEM, f"{convo}\nuser: {question}", max_tokens=80).strip()
    return rewritten or question


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

    # ---- retrieval ------------------------------------------------------------
    def retrieve(self, query: str, paper_ids=None):
        k = 20 if settings.use_rerank else settings.top_k
        if settings.use_hybrid:
            from .hybrid import hybrid_search
            hits = hybrid_search(self.store, query, paper_ids, k=k)   # fused hits have score 0
        else:
            hits = [h for h in self.store.search(query, k=k, paper_ids=paper_ids)
                    if h["score"] >= settings.min_score]              # no fallback to weak hits
        if hits and settings.use_rerank:
            from .rerank import rerank
            hits = rerank(query, hits, top_n=settings.top_k)
        return hits

    # ---- question answering ---------------------------------------------------
    def ask(self, question: str, paper_ids=None, history=None) -> dict:
        section = detect_section(question)
        query = standalone(history, question)

        hits = self.store.get_section_chunks(section, paper_ids) if section else []
        if not hits:
            hits = self.retrieve(query, paper_ids)
        if not hits and query != question:
            hits = self.retrieve(question, paper_ids)
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "invalid_citations": [], "query": query}

        note = (f"The user asks specifically about the {section} section. Use that section's "
                "evidence when available; if it lacks the requested information, say so.\n\n"
                if section else "")
        user = (f"{note}CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}\n\n"
                "Answer directly, use only the supplied evidence, and cite with [n]. "
                "If the evidence is insufficient, state what could not be found.")
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
                f"ASPECTS: {', '.join(aspects)}\nProduce the comparison with citations.")
        answer = complete(COMPARE_SYSTEM, user, max_tokens=1800)
        _, invalid = parse_citations(answer, hits)
        return {"answer": answer, "sources": hits, "invalid_citations": invalid}
