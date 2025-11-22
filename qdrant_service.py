# query_service.py
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from qdrant_client import QdrantClient

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    top_k: int = None

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

qcfg = cfg["qdrant"]
COLLECTION = qcfg["collection"]
embed_model_name = cfg["models"]["embedding"]
gen_model_name = cfg["models"]["generation"]

embedder = SentenceTransformer(embed_model_name)
generator = pipeline("text-generation", model=gen_model_name)   # swap for a more capable model
q = QdrantClient(url=qcfg["url"], api_key=qcfg.get("api_key"))

@app.post("/query")
def query(req: QueryRequest):
    top_k = req.top_k or cfg["ingest"]["top_k"]

    # embed the query
    q_vector = embedder.encode(req.question).tolist()

    # retrieve from Qdrant
    hits = q.search(collection_name=COLLECTION, query_vector=q_vector, limit=top_k)
    contexts = []
    for h in hits:
        payload = h.payload
        contexts.append(f"(source:{payload.get('source_file')} page_chunk:{payload.get('chunk_index')}) {payload.get('text')}")

    # build prompt and generate
    prompt = "You are a helpful assistant. Use the following context to answer the question.\n\n"
    prompt += "\n\n---\n\n".join(contexts)
    prompt += f"\n\nQuestion: {req.question}\nAnswer:"

    # You might want to use an instruction model or an LLM with longer context
    out = generator(prompt, max_length=256, do_sample=False)
    answer = out[0]["generated_text"]

    return {"answer": answer, "retrieved": [h.payload for h in hits]}

# Run: uvicorn query_service:app --reload --port 8000
