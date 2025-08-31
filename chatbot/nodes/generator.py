from datetime import datetime
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from chatbot.state import GraphState
from chatbot.prompts import GENERATOR_SYSTEM_PROMPT

class GeneratorNode:
    def __init__(self, chat_model, temperature):
        self.model = ChatOpenAI(model=chat_model, temperature=temperature)

    def process(self, state: GraphState) -> Dict[str, Any]:
        template = ChatPromptTemplate.from_messages(
            [
                ("system", GENERATOR_SYSTEM_PROMPT),
                *state["history"],
                ("user", "Context:\n{context}\n\nQuestion: {question}\n\nCurrent Datetime: {current_datetime}"),
            ]
        )
        context_text = "\n\n".join(
            f"Source: {d.metadata.get('source', '?')}\n{d.page_content}" for d in state["contexts"]
        )
        prompt = template.format_messages(context=context_text,
                                          question=state["question"],
                                          current_datetime=str(datetime.now())
                                          )

        resp = self.model.invoke(prompt)
        answer = resp.content.strip()
        history = state["history"] + [("user", state["question"]), ("assistant", answer)]

        return {
            "answer": answer,
            "history": history,
        }