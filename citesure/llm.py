"""The ONLY file that talks to the LLM. Swap providers by rewriting these two functions."""
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()         # reads ANTHROPIC_API_KEY from the environment
    return _client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def complete(system: str, user: str, max_tokens: int = 1200) -> str:
    msg = client().messages.create(
        model=settings.llm_model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if b.type == "text")


def stream(system: str, user: str, max_tokens: int = 1200):
    with client().messages.stream(
            model=settings.llm_model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}]) as s:
        yield from s.text_stream
