import os
import streamlit as st
from dotenv import load_dotenv
from chatbot import ChatGraph

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "index/faiss_index")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
GUARDRAIL_TEMPERATURE = float(os.getenv("GUARDRAIL_TEMPERATURE", 0))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 4))
RETRIEVER_THRESHOLD = float(os.getenv("RETRIEVER_THRESHOLD", 0.25))
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", 0))


graph = ChatGraph(guardrail_model=CHAT_MODEL,
                  guardrail_temperature=GUARDRAIL_TEMPERATURE,
                  retriever_index_dir=INDEX_DIR,
                  retriever_embed_model=EMBED_MODEL,
                  retriever_k=RETRIEVER_K,
                  retriever_threshold=RETRIEVER_THRESHOLD,
                  generator_model=CHAT_MODEL,
                  generator_temperature=GENERATOR_TEMPERATURE)


st.set_page_config(page_title="Nana-bot", page_icon="🧠", layout="centered")
st.title("🧠 Nana-bot")
st.caption("Meet Nana-bot: your shortcut to knowing Nana inside out.")


if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Type your question about Nana…")
if q:
    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        answer = graph.process(q, st.session_state.history)

        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
