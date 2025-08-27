# 🧠 Nana-bot

A small Streamlit app that answers only questions about **Nana Karapetyan** using a guardrail + RAG setup.

---

## 🚀 How to Run

### 1. **Clone this repo**
   ```bash
   https://github.com/NanaK99/nana_bot.git
   cd nana-bot
   ```
  
## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Prepare your data

- Put your private files (e.g. cv.pdf, fun_facts.txt) inside the data/ folder.
- The repo only contains data/README.md as a placeholder.
- Your real documents are ignored for privacy reasons.

## 4. Build the vector index
```bash
python ingest.py
```

## 5. Run the app
```bash
streamlit run app.py
```

## 6. Open in browser
Streamlit will print a local URL, usually http://localhost:8501.
Open it and start chatting with Nana-bot 🎉