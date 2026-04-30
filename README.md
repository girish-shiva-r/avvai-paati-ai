# Avvaiyar Paatti AI 👵

A full-stack AI application that dispenses ancient Tamil wisdom from the *Aathichoodi* to solve modern problems.

## Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI
* **RAG Pipeline:** FAISS, Hugging Face Embeddings, custom fine-tuned `xlm-roberta` model
* **LLM:** Google Gemini 2.0 Flash

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Add your `.env` file with your `HF_TOKEN` and `GOOGLE_API_KEY`.
3. Start the backend: `uvicorn main:app --reload`
4. Start the frontend: `streamlit run app.py`
