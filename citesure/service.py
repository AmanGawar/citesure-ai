
"""CiteSure: shared service for the UI and API."""
import re

from .config import settings
from .ingest import ingest_pdf
from .llm import complete
from .prompts import (
    COMPARE_SYSTEM,
    FACETS,
    NOT_FOUND,
    SYSTEM,
    format_context,
    parse_citations,
)
from .store import PaperStore

REWRITE_SYSTEM = (
    "Rewrite the user's last question as a standalone search query, "
    "resolving pronouns using the conversation. Output only the query."
)

SECTION_PATTERNS = {
    "abstract": r"\babstract\b|\basbstract\b|\babstarct\b",
    "introduction": r"\bintroduction\b|\bbackground\b",
    "methodology": r"\bmethodology\b|\bmethods?\b|\bapproach\b",
    "results": r"\bresults?\b|\bfindings\b|\bexperiments?\b",
    "limitations": r"\blimitations?\b|\bweaknesses\b",
    "conclusion": r"\bconclusions?\b|\bconcluding remarks\b",
}


def detect_section(question: str):
    q = question.lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, q):
            return section
    return None


def standalone(history, question: str) -> str:
    if not history:
        return question

    convo = "\n".join(
        f"{role}: {text}" for role, text in history[-6:]
    )
    rewritten = complete(
        REWRITE_SYSTEM,
        f"{convo}\nuser: {question}",
        max_tokens=80,
    ).strip()

    return rewritten or question


class CiteSure:
    def __init__(self):
        self.store = PaperStore()

    # ---- indexing ---------------------------------------------------------
    def add_pdf(self, path: str) -> dict:
        chunks = ingest_pdf(path)
        if not chunks:
            raise ValueError(
                "No extractable text found. Is this a scanned PDF?"
            )

        pid = chunks[0].paper_id
        if not self.store.has_paper(pid):
            self.store.add_chunks(chunks)

        return {
            "paper_id": pid,
            "title": chunks[0].title,
            "chunks": len(chunks),
        }

    # ---- retrieval --------------------------------------------------------
    def retrieve(self, query: str, paper_ids=None):
        k = max(settings.top_k, 8)
        collected = {}

        queries = [query]

        # Alternate query wording can help semantic retrieval.
        if query.lower().strip() != query:
            queries.append(query.lower().strip())

        for q in queries:
            try:
                if settings.use_hybrid:
                    from .hybrid import hybrid_search
                    hits = hybrid_search(
                        self.store, q, paper_ids, k=k
                    )
                else:
                    hits = self.store.search(
                        q, k=k, paper_ids=paper_ids
                    )
            except Exception:
                continue

            for hit in hits:
                collected[hit["id"]] = hit

        hits = list(collected.values())

        if not hits:
            return []

        # Do not apply a cosine threshold to hybrid results:
        # hybrid search may return fused scores of zero.
        if not settings.use_hybrid:
            filtered = [
                h for h in hits
                if h.get("score", 0) >= settings.min_score
            ]
            if filtered:
                hits = filtered

        if settings.use_rerank and hits:
            from .rerank import rerank
            hits = rerank(query, hits, top_n=settings.top_k)

        return hits

    # ---- question answering ----------------------------------------------
    def ask(self, question: str, paper_ids=None, history=None) -> dict:
        section = detect_section(question)
        query = standalone(history, question)

        hits = []

        # Exact section retrieval for questions like "What is the abstract?"
        if section:
            hits = self.store.get_section_chunks(
                section, paper_ids=paper_ids
            )

        # Fall back to semantic/hybrid retrieval if section retrieval fails.
        if not hits:
            hits = self.retrieve(query, paper_ids)

        # Final fallback: search using the original question.
        if not hits and query != question:
            hits = self.retrieve(question, paper_ids)

        if not hits:
            return {
                "answer": NOT_FOUND,
                "sources": [],
                "invalid_citations": [],
                "query": query,
            }

        instruction = ""
        if section:
            instruction = (
                f"The user asks specifically about the {section} "
                "section. Use that section's evidence when available. "
                "If it does not contain the requested information, "
                "say so instead of inventing details."
            )

        user = (
            f"{instruction}\n\n"
            f"CONTEXT:\n{format_context(hits)}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer the user's actual question directly. "
            "Use only the supplied evidence. Include citations "
            "using the required [n] format. If the evidence is "
            "insufficient, clearly state what could not be found."
        )

        answer = complete(SYSTEM, user)
        _, invalid = parse_citations(answer, hits)

        return {
            "answer": answer,
            "sources": hits,
            "invalid_citations": invalid,
            "query": query,
        }

    # ---- summarization ----------------------------------------------------
    def summarize(self, paper_id: str) -> dict:
        sections = []

        for name, question in FACETS.items():
            hits = self.store.search(
                question, k=4, paper_ids=[paper_id]
            )
            user = (
                f"CONTEXT:\n{format_context(hits)}\n\n"
                f"QUESTION: {question}\n"
                "Answer in 2-4 sentences."
            )
            answer = complete(SYSTEM, user, max_tokens=500)
            sections.append((name, answer, hits))

        digest = "\n\n".join(
            f"{name}: {answer}"
            for name, answer, _ in sections
        )

        tldr = complete(
            "Write a 3-sentence TL;DR of a research paper "
            "from these notes. Do not include citation numbers.",
            digest,
            max_tokens=300,
        )

        return {"tldr": tldr, "sections": sections}

    # ---- comparison -------------------------------------------------------
    def compare(self, paper_ids, aspects) -> dict:
        found = {}

        for aspect in aspects:
            for pid in paper_ids:
                for hit in self.store.search(
                    aspect, k=3, paper_ids=[pid]
                ):
                    found[hit["id"]] = hit

        hits = list(found.values())
        titles = sorted({
            h["meta"]["title"] for h in hits
        })

        user = (
            f"CONTEXT:\n{format_context(hits)}\n\n"
            f"PAPERS: {'; '.join(titles)}\n"
            f"ASPECTS: {', '.join(aspects)}\n"
            "Produce the comparison with evidence-based citations."
        )

        answer = complete(
            COMPARE_SYSTEM, user, max_tokens=1800
        )
        _, invalid = parse_citations(answer, hits)

        return {
            "answer": answer,
            "sources": hits,
            "invalid_citations": invalid,
        }
