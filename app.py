import streamlit as st
from dotenv import load_dotenv
from graph_app import build_graph

load_dotenv()

st.set_page_config(page_title="Nana-bot", page_icon="🧠", layout="centered")
st.title("🧠 Nana-bot")
st.caption("Meet Nana-bot: your shortcut to knowing Nana inside out.")

graph = build_graph()

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
        result = graph.invoke({"question": q,
                               "history": st.session_state.history,
                               "contexts": [], "answer": None})
        ans = result["answer"]

        st.markdown(ans)

        contexts = result.get("contexts", [])
        good = [(d, s) for (d, s) in contexts if s >= 0.25]

    st.session_state.history.append(("assistant", ans))
