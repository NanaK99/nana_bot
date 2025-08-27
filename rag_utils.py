import faiss
import tiktoken
import numpy as np
import os, re, json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from openai import OpenAI
from pypdf import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()
INDEX_DIR = "index/faiss_index"

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def get_client() -> OpenAI:
    return OpenAI()

def read_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        return ""

def read_text_files(folder: str) -> list[tuple[str, str]]:
    """Return list of (source_path, text). Supports .txt, .pdf."""
    out = []
    for p in Path(folder).glob("**/*"):
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            out.append((str(p), p.read_text(encoding="utf-8", errors="ignore")))
        elif ext == ".pdf":
            text = read_pdf_text(str(p))
            if text.strip():
                out.append((str(p), text))
    return out

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 50, model: str="gpt-4o-mini") -> List[str]:
    enc = tiktoken.encoding_for_model(model if "gpt" in model else "gpt-4o-mini")
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        chunk = enc.decode(toks[i:j])
        chunks.append(clean_text(chunk))
        i = j - overlap_tokens
        if i < 0: i = 0
    return [c for c in chunks if c]

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def build_faiss_index(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    nv = vectors / norms
    index = faiss.IndexFlatIP(nv.shape[1])
    index.add(nv)
    return index

def save_index(index, meta: List[Dict], folder="index"):
    Path(folder).mkdir(exist_ok=True)
    faiss.write_index(index, f"{folder}/index.faiss")
    with open(f"{folder}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index(folder="index"):
    index = faiss.read_index(f"{folder}/index.faiss")
    with open(f"{folder}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def search(client: OpenAI, index, meta, query: str, top_k=4) -> List[Dict]:
    qv = embed_texts(client, [query])
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-10)
    D, I = index.search(qv, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    out = []
    for s, i in zip(scores, idxs):
        if i == -1: continue
        out.append({**meta[i], "score": float(s)})
    return out

def load_vectorstore() -> FAISS:
    vs = FAISS.load_local(
        INDEX_DIR,
        embeddings=OpenAIEmbeddings(model=EMBED_MODEL),
        allow_dangerous_deserialization=True,
    )
    return vs