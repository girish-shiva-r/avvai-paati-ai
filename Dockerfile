# Hugging Face Docker Space (backend API). Add GOOGLE_API_KEY as a Space secret.
FROM python:3.11-slim
WORKDIR /code
RUN useradd -m -u 1000 user
ENV HF_HOME=/code/.hf PAATTI_EMBEDDER=mpnet-ml PAATTI_LLM=gemini-3.5-flash-lite PORT=7860
# CPU-only torch keeps the image ~2 GB smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# bake the embedding model into the image so cold starts don't download 2 GB
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')" && chown -R user /code/.hf
COPY --chown=user paatti/ paatti/
COPY --chown=user data/aathichoodi_kb.json data/
COPY --chown=user api.py .
USER user
EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
