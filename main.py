from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.rag_pipeline import RAGPipeline

app = FastAPI(
    title="E-Commerce RAG API",
    description="Retrieval Augmented Generation for E-Commerce using Endee Vector DB",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    global rag
    rag = RAGPipeline()
    await rag.initialize()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class IngestRequest(BaseModel):
    force_reingest: bool = False


@app.get("/")
def root():
    return {
        "message": "E-Commerce RAG API powered by Endee Vector DB",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_products(req: IngestRequest):
    """Ingest sample e-commerce product data into Endee vector index."""
    count = await rag.ingest_products(force=req.force_reingest)
    return {"status": "success", "products_ingested": count}


@app.post("/query")
async def query(req: QueryRequest):
    """
    Answer a natural-language question about products using RAG.
    Retrieves relevant product context from Endee, then generates an answer via LLM.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = await rag.answer(req.question, top_k=req.top_k)
    return result


@app.get("/products/search")
async def search_products(q: str, top_k: int = 5):
    """Pure semantic vector search — returns matching products without LLM generation."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = await rag.semantic_search(q, top_k=top_k)
    return {"query": q, "results": results}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
