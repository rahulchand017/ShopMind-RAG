import os
import json
import asyncio
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
import httpx

ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INDEX_NAME = "ecommerce_products"
EMBEDDING_DIM = 384


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        token = ENDEE_TOKEN if ENDEE_TOKEN else None
        self.client = Endee(token)
        self.client.set_base_url(f"{ENDEE_HOST}/api/v1")
        self.index = None

    async def initialize(self):
        """Create Endee index if it doesn't exist."""
        try:
            self.index = self.client.get_index(INDEX_NAME)
            print(f"[Endee] Connected to existing index: {INDEX_NAME}")
        except Exception:
            self.client.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
            self.index = self.client.get_index(INDEX_NAME)
            print(f"[Endee] Created new index: {INDEX_NAME}")

    async def ingest_products(self, force: bool = False) -> int:
        """Load products from JSON, embed and upsert into Endee."""
        products = self._load_products()

        vectors = []
        for product in products:
            text = self._product_to_text(product)
            embedding = self.embedder.encode(text).tolist()
            vectors.append({
                "id": str(product["id"]),
                "vector": embedding,
                "meta": {
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "rating": product["rating"],
                    "description": product["description"],
                    "brand": product["brand"],
                    "in_stock": product["in_stock"],
                }
            })

        self.index.upsert(vectors)
        print(f"[Endee] Upserted {len(vectors)} product vectors.")
        return len(vectors)

    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Embed the query and retrieve top-k products from Endee."""
        query_vec = self.embedder.encode(query).tolist()
        results = self.index.query(vector=query_vec, top_k=top_k)
        output = []
        for r in results:
            output.append({
                "id": r.id,
                "similarity": round(r.similarity, 4),
                **r.meta
            })
        return output

    async def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Full RAG: retrieve context from Endee, generate answer with LLM."""
        # 1. Retrieve
        context_items = await self.semantic_search(question, top_k=top_k)

        # 2. Build context string
        context_str = self._build_context(context_items)

        # 3. Generate answer
        if OPENAI_API_KEY:
            answer_text = await self._generate_openai(question, context_str)
        else:
            answer_text = self._generate_local(question, context_items)

        return {
            "question": question,
            "answer": answer_text,
            "retrieved_products": context_items,
            "source": "openai" if OPENAI_API_KEY else "rule-based"
        }

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _product_to_text(self, product: Dict) -> str:
        return (
            f"{product['name']} by {product['brand']}. "
            f"Category: {product['category']}. "
            f"Price: ${product['price']}. "
            f"Rating: {product['rating']}/5. "
            f"{product['description']}"
        )

    def _build_context(self, items: List[Dict]) -> str:
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(
                f"{i}. {item['name']} (${item['price']}) — {item['description']} "
                f"[Rating: {item['rating']}/5, Brand: {item['brand']}, "
                f"In Stock: {'Yes' if item['in_stock'] else 'No'}]"
            )
        return "\n".join(lines)

    def _generate_local(self, question: str, items: List[Dict]) -> str:
        """Simple rule-based answer when no LLM key is set."""
        if not items:
            return "Sorry, no relevant products found."
        top = items[0]
        return (
            f"Based on your query, the most relevant product is **{top['name']}** "
            f"by {top['brand']}, priced at ${top['price']} with a {top['rating']}/5 rating. "
            f"{top['description']} "
            f"{'It is currently in stock.' if top['in_stock'] else 'It is currently out of stock.'}"
        )

    async def _generate_openai(self, question: str, context: str) -> str:
        """Call OpenAI chat completion for answer generation."""
        system_prompt = (
            "You are a helpful e-commerce assistant. "
            "Use the product catalog context below to answer the customer's question. "
            "Be concise, friendly, and specific."
        )
        user_prompt = (
            f"Product Catalog Context:\n{context}\n\n"
            f"Customer Question: {question}"
        )
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    def _load_products(self) -> List[Dict]:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
        with open(path, "r") as f:
            return json.load(f)
