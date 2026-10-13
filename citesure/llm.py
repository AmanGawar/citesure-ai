import os
import streamlit as st
from groq import Groq

_client = None

def client():
    global _client

    if _client is None:
        key = os.getenv("GROQ_API_KEY")

        if not key:
            key = st.secrets["GROQ_API_KEY"]

        _client = Groq(api_key=key.strip())

    return _client
