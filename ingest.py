import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index/faiss_index"))
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def load_docs() -> list[Document]:
    docs: list[Document] = []
    for p in DATA_DIR.glob("**/*"):
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix.lower() == ".pdf":
            text = read_pdf(p)
        else:
            continue
        text = " ".join(text.split())
        if not text.strip():
            continue
        docs.append(Document(page_content=text, metadata={"source": str(p)}))
    return docs

def main():
    raw_docs = load_docs()
    if not raw_docs:
        raise SystemExit("No docs in ./data (txt/md/pdf). Add your CV + fun_facts.txt.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embedding=embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"Saved {len(chunks)} chunks to {INDEX_DIR}")

if __name__ == "__main__":
    main()
