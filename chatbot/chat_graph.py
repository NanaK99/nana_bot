from chatbot.state import GraphState
from chatbot.nodes import (GuardrailNode,
                           RetrieverNode,
                           GeneratorNode)

from langgraph.graph import StateGraph, END


class ChatGraph:
    def __init__(self, guardrail_model,
                 guardrail_temperature,
                 retriever_index_dir,
                 retriever_embed_model,
                 retriever_k,
                 retriever_threshold,
                 generator_model,
                 generator_temperature):

        self._guardrail_node = GuardrailNode(chat_model=guardrail_model, temperature=guardrail_temperature)
        self._retriever_node = RetrieverNode(index_dir=retriever_index_dir, embed_model=retriever_embed_model, k=retriever_k, threshold=retriever_threshold)
        self._generate_node = GeneratorNode(chat_model=generator_model, temperature=generator_temperature)

        self._graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(GraphState)

        g.add_node("guardrail", self._guardrail_node.process)
        g.add_node("retrieve", self._retriever_node.process)
        g.add_node("generate", self._generate_node.process)

        g.set_entry_point("guardrail")
        g.add_conditional_edges("guardrail",
                                lambda state: END if state["answer"] else "retrieve")

        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)

        return g.compile()

    def process(self, question, history):
        result = self._graph.invoke({"question": question,
                               "history": history,
                               "contexts": [], "answer": None})
        answer = result["answer"]

        return answer