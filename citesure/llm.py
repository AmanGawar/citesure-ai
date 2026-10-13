
import os
import streamlit as st
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

_client = None


def client():
    global _client

    if _client is None:
        key = os.getenv("GROQ_API_KEY")

        if not key:
            key = st.secrets.get("GROQ_API_KEY", "")

        if not key.strip():
            raise RuntimeError("GROQ_API_KEY is missing from Streamlit Secrets.")

        _client = Groq(api_key=key.strip())

    return _client


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def complete(system: str, user: str, max_tokens: int = 1200) -> str:
    response = client().chat.completions.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def stream(system: str, user: str, max_tokens: int = 1200):
    response = client().chat.completions.create(
        model=settings.llm_model,
        max_tokens=max_tokens,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
