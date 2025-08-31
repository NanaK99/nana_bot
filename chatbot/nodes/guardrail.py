from typing import Dict, Any
from langchain_openai import ChatOpenAI

from chatbot.state import GraphState
from chatbot.prompts import GUARDRAIL_SYSTEM_PROMPT

class GuardrailNode:
    def __init__(self, chat_model, temperature):
        self.model = ChatOpenAI(model=chat_model, temperature=temperature)

    def process(self, state: GraphState) -> Dict[str, Any]:
        prompt = [
            ("system", GUARDRAIL_SYSTEM_PROMPT),
            *state["history"],
            ("user", state["question"])
        ]

        resp = self.model.invoke(prompt)
        answer = resp.content

        if answer != "PASS":
            history = state["history"] + [("user", state["question"]), ("assistant", answer)]

            return {
                "answer": answer,
                "history": history,
            }
        else:
            return {"answer": None}