import os
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Tuple, Dict, Any

from rag_utils import load_vectorstore

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

class GraphState(TypedDict):
    question: str
    history: List[Tuple[str, str]]
    contexts: List[Tuple[Document, float]]
    answer: Optional[str]


def guardrail(state: GraphState) -> Dict[str, Any]:
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

    SYSTEM = (
        """
        You are a filter. Only allow questions about the person Nana Karapetyan.  

        - If the question is not about Nana (it is about people who are unrelated to her or other topics, such as asking general info about anything), reply with a polite message informing that the question is out of scope and you cannot answer
        - If the question is about her, related to her (such as her husband, family, or any other info related to her), or you are not 100% sure, reply exactly with: "PASS" 
        - If the message is just greeting, saying thanks, or reply accordingly
        - If the message is "I want to talk about Nana" or similar, reply with: "Of course! I’d love to talk about Nana. What would you like to know?"
        """
    )

    prompt = [
        ("system", SYSTEM),
        *state["history"],
        ("user", state["question"])
    ]


    resp = model.invoke(prompt)
    answer = resp.content.strip()

    if answer != "PASS":
        history = state["history"] + [("user", state["question"]), ("assistant", answer)]

        return {
            "answer": answer,
            "history": history,
        }
    else:
        return {"answer": None}


def retrieve(state: GraphState) -> Dict[str, Any]:
    vs = load_vectorstore()
    q = state["question"]

    docs_scores = vs.similarity_search_with_score(q, k=4)
    return {"question": q, "contexts": docs_scores, "answer": state.get("answer")}


def generate(state: GraphState) -> Dict[str, Any]:
    model = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

    threshold = 0.25
    good_pairs = [(d, s) for (d, s) in state["contexts"] if s >= threshold]
    good_docs = [d for (d, _) in good_pairs]

    SYSTEM = (
        """
        You are a helpful, precise assistant that only answers about the person
        described in the provided context. 
        
        If the answer is not in context, say exactly:
        "I don’t know her well yet, but she’s kind and will gladly answer your questions herself".
        
        Do not invent facts.
        """
    )

    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            *state["history"],
            ("user", "Context:\n{context}\n\nQuestion: {question}\n\nCurrent Datetime: {current_datetime}"),
        ]
    )
    context_text = "\n\n".join(
        f"Source: {d.metadata.get('source', '?')}\n{d.page_content}" for d in good_docs
    )
    prompt = template.format_messages(context=context_text,
                                      question=state["question"],
                                      current_datetime=str(datetime.now())
                                      )

    resp = model.invoke(prompt)
    answer = resp.content.strip()
    history = state["history"] + [("user", state["question"]), ("assistant", answer)]

    return {
        "answer": answer,
        "history": history,
    }


def guardrail_condition(state: GraphState) -> str:
    if state["answer"]:
        return END
    else:
        return "retrieve"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("guardrail", guardrail)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.set_entry_point("guardrail")
    g.add_conditional_edges("guardrail", guardrail_condition)

    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()
