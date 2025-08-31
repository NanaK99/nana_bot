from typing import Dict, Any, Annotated, Optional
from langchain_openai import ChatOpenAI

from chatbot.state import GraphState
from chatbot.prompts import GUARDRAIL_SYSTEM_PROMPT
from typing import TypedDict

class GuardrailOutput(TypedDict):
    passed: Annotated[bool, "Whether message is about Nana"]
    answer: Annotated[str, "Polite reply in case message is out of scope"]
    reason: Annotated[Optional[str], "Reason why the message is out of scope"]

class GuardrailNode:
    def __init__(self, chat_model, temperature):
        self.model = ChatOpenAI(model=chat_model, temperature=temperature).with_structured_output(GuardrailOutput)

    def process(self, state: GraphState) -> Dict[str, Any]:
        prompt = [
            ("system", GUARDRAIL_SYSTEM_PROMPT),
            *state["history"],
            ("user", state["question"])
        ]

        resp = self.model.invoke(prompt)
        answer = resp["answer"]

        if not resp["passed"]:
            history = state["history"] + [("user", state["question"]), ("assistant", answer)]

            return {
                "guardrail_passed": False,
                "answer": answer,
                "history": history,
            }
        else:
            return {"guardrail_passed": True, "answer": None}