from langchain_core.documents import Document
from typing import TypedDict, List, Optional, Tuple

class GraphState(TypedDict):
    question: str
    history: List[Tuple[str, str]]
    contexts: List[Document]
    answer: Optional[str]