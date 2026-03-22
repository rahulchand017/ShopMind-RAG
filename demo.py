#!/usr/bin/env python3
"""
Quick demo script — ingests products and runs sample RAG queries.
Run after starting the API: python scripts/demo.py
"""

import httpx
import json

BASE_URL = "http://localhost:8000"

def header(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def main():
    client = httpx.Client(base_url=BASE_URL, timeout=60)

    # 1. Health check
    header("1. Health Check")
    r = client.get("/health")
    print(r.json())

    # 2. Ingest products
    header("2. Ingesting Products into Endee")
    r = client.post("/ingest", json={"force_reingest": True})
    print(json.dumps(r.json(), indent=2))

    # 3. Semantic search
    header("3. Semantic Search: 'good headphones for travel'")
    r = client.get("/products/search", params={"q": "good headphones for travel", "top_k": 3})
    data = r.json()
    for p in data["results"]:
        print(f"  [{p['similarity']:.4f}] {p['name']} — ${p['price']}")

    # 4. RAG Query 1
    header("4. RAG Query: 'I need something to listen to music quietly'")
    r = client.post("/query", json={"question": "I need something to listen to music quietly", "top_k": 3})
    data = r.json()
    print(f"Answer: {data['answer']}")
    print(f"Source: {data['source']}")

    # 5. RAG Query 2
    header("5. RAG Query: 'What running shoes do you have?'")
    r = client.post("/query", json={"question": "What running shoes do you have?", "top_k": 3})
    data = r.json()
    print(f"Answer: {data['answer']}")

    # 6. RAG Query 3
    header("6. RAG Query: 'Best laptop under $1200 for students'")
    r = client.post("/query", json={"question": "Best laptop under $1200 for students", "top_k": 3})
    data = r.json()
    print(f"Answer: {data['answer']}")

    print("\n✅ Demo completed successfully!\n")

if __name__ == "__main__":
    main()
