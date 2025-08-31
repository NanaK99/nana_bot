from typing import Dict, Any

from chatbot.state import GraphState
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


class RetrieverNode:
    def __init__(self, index_dir, embed_model, k, threshold):
        self.vector_store = FAISS.load_local(
            index_dir,
            embeddings=OpenAIEmbeddings(model=embed_model),
            allow_dangerous_deserialization=True,
        )

        self.k = k
        self.threshold = threshold

    def process(self, state: GraphState) -> Dict[str, Any]:
        q = state["question"]

        docs_with_scores = self.vector_store.similarity_search_with_score(q, k=self.k)

        filtered_docs = [d for (d, s) in docs_with_scores if s >= self.threshold]

        return {"contexts": filtered_docs}
