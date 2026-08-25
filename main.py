import os
import json
import random
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# ==========================================
# 1. SETUP & CUSTOM MODEL LOADING
# ==========================================
load_dotenv()
app = FastAPI(title="Avvaiyar Paatti Brain 🧠")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "GSR-608001/aathichoodi_pro_model"
MODEL_CACHE_DIR = "./cached_model_fixed"

# Global Storage
RETRIEVER = None
DF = None

def ensure_custom_model_integrity():
    """Forces the use of YOUR Fine-Tuned Model by repairing it if needed."""
    print("🧠 Brain Status: Validating Custom Model...")
    
    # 1. Download if missing
    if not os.path.exists(MODEL_CACHE_DIR) or not os.path.exists(os.path.join(MODEL_CACHE_DIR, "model.safetensors")):
        print("⬇️ Downloading Fine-Tuned Model (Critical for accuracy)...")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                local_dir=MODEL_CACHE_DIR,
                token=HF_TOKEN,
                ignore_patterns=["*.msgpack", "*.h5"]
            )
        except Exception as e:
            print(f"❌ CRITICAL: Custom Model Download Failed. {e}")
            raise e

    # 2. Fix Config (The Self-Repair)
    config_path = os.path.join(MODEL_CACHE_DIR, "config.json")
    try:
        with open(config_path, "r") as f: config = json.load(f)
        if "model_type" not in config:
            print("🔧 Injecting missing metadata into config...")
            config["model_type"] = "xlm-roberta"
            with open(config_path, "w") as f: json.dump(config, f, indent=2)
    except Exception as e:
        print(f"⚠️ Config Warning: {e}")

@app.on_event("startup")
async def startup_event():
    global RETRIEVER, DF
    
    # 1. Prepare Model
    ensure_custom_model_integrity()
    
    # 2. Load Embeddings (Strictly Custom)
    print("🔌 Loading Fine-Tuned Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_CACHE_DIR,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 3. Load Data
    print("📂 Loading Knowledge Base...")
    paths = [
        "aathichoodi_english_rich (1).csv",
        "aathichoodi_english_rich.csv", 
        "/Users/girishshivar/Documents/Tamizhakaran/aathichoodi_english_rich.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            DF = pd.read_csv(p)
            break
            
    if DF is None: raise RuntimeError("❌ CSV File Missing!")

    documents = []
    for _, row in DF.iterrows():
        content = row.get('Embedding_Text', row['Rich_English_Explanation'])
        meta = {"verse_no": row['Verse_No'], "verse": row['Verse'], "eng": row['Original_English']}
        documents.append(Document(page_content=content, metadata=meta))

    # 4. Build Vector DB
    vectorstore = FAISS.from_documents(documents, embeddings)
    RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ Brain is Online: Custom Model Loaded.")

# ==========================================
# 2. THE PERSONA LOGIC
# ==========================================
class QueryRequest(BaseModel):
    query: str

def get_api_key():
    keys = [os.getenv(f"GOOGLE_API_KEY_{i}") for i in range(1, 4) if os.getenv(f"GOOGLE_API_KEY_{i}")]
    if not keys: keys = [os.getenv("GOOGLE_API_KEY")]
    return random.choice(keys)

def get_rag_chain(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=api_key)
    template = """
    You are 'Avvaiyar Paatti' (Grandma Avvai), a loving and wise Tamil grandmother.
    The user is your grandchild who has come to you with a problem: "{question}"

    Here are the verses from Aathichoodi that might help:
    {context}

    INSTRUCTIONS:
    1.  **Analyze & Select:** Read all the verses. Pick the ONE verse that directly solves their specific trouble.
    2.  **Affectionate Tone:** Start with "En Chellame" (My darling) or "Kanna". Make them feel safe and loved immediately.
    3.  **The Wisdom:** Quote that one best Tamil verse and its Number.
    4.  **The Advice:** Weave the meaning of that verse into a gentle, matured advice. Use the other verses only if they add depth, but keep the focus on the main solution.
    5.  **Goal:** Make them feel relieved and morally guided. Speak to their heart, not their brain.
    6.  **Format:** Keep it crisp (Max 5-6 lines). Use soothing emojis (👵, 🌿, ✨, ❤️).

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": RETRIEVER | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    if not RETRIEVER: raise HTTPException(status_code=503, detail="Brain loading...")
    try:
        api_key = get_api_key()
        chain = get_rag_chain(api_key)
        return {"response": chain.invoke(request.query)}
    except Exception as e:
        return {"response": f"👵 Paatti needs a moment to rest. (Error: {str(e)})"}

@app.post("/chat_audio")
async def chat_audio_endpoint(file: UploadFile = File(...)):
    if not RETRIEVER: raise HTTPException(status_code=503, detail="Brain loading...")
    
    try:
        api_key = get_api_key()
        audio_bytes = await file.read()
        
        # 1. Transcribe Audio using Gemini Native SDK
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt_stt = "Transcribe the speech in this audio accurately. Return ONLY the transcribed text, nothing else."
        
        response_stt = model.generate_content([
            prompt_stt,
            {"mime_type": file.content_type or "audio/wav", "data": audio_bytes}
        ])
        transcribed_text = response_stt.text.strip()
        
        # 2. Run the transcribed text through the existing RAG pipeline
        chain = get_rag_chain(api_key)
        paatti_response = chain.invoke(transcribed_text)
        
        return {
            "transcribed_text": transcribed_text,
            "response": paatti_response
        }
    except Exception as e:
        return {"response": f"👵 Paatti needs a moment to rest. (Audio Error: {str(e)})"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
