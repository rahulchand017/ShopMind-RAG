# 🛒 E-Commerce RAG with Endee Vector DB

A production-ready **Retrieval Augmented Generation (RAG)** system for e-commerce, powered by [Endee](https://github.com/endee-io/endee) as the high-performance vector database.

Ask natural-language questions like *"What's a good laptop for students?"* and get accurate, context-grounded answers backed by a live product catalog — all in milliseconds.

---

## 📌 Problem Statement

E-commerce platforms often struggle with product discovery. Traditional keyword search fails when customers use natural language or describe features rather than exact product names. This project solves that by:

- Embedding product descriptions as semantic vectors stored in Endee
- Retrieving the most relevant products using cosine similarity vector search
- Generating a natural, helpful answer using a language model (OpenAI) or a rule-based fallback

---

## 🏗️ System Design

```
Customer Question
       │
       ▼
┌──────────────────┐
│  FastAPI Server  │  ← REST API (Python)
└──────────────────┘
       │
       ▼ embed with SentenceTransformer (all-MiniLM-L6-v2)
┌──────────────────┐
│   Endee (nDD)    │  ← Vector DB on port 8080
│  cosine / INT8   │    returns top-K products
└──────────────────┘
       │
       ▼ build context string
┌──────────────────┐
│  LLM / Fallback  │  ← OpenAI GPT-3.5 (optional)
└──────────────────┘
       │
       ▼
  Final Answer + Retrieved Products
```

### Key Components

| Component | Role |
|---|---|
| **Endee** | Vector database — stores & retrieves product embeddings |
| **SentenceTransformers** | Converts product text & queries into 384-dim vectors |
| **FastAPI** | REST API layer exposing `/ingest`, `/query`, `/products/search` |
| **OpenAI GPT-3.5** | Optional LLM for generating fluent answers from retrieved context |

---

## 🔍 How Endee Is Used

Endee is the **core retrieval engine** of this project.

1. **Index Creation** — A `cosine` index with `INT8` precision and 384 dimensions is created in Endee at startup.
2. **Upserting Vectors** — Each product is converted to text, embedded, and stored as a vector with metadata (name, price, brand, etc.) via `index.upsert()`.
3. **Vector Search** — Customer queries are embedded and searched against all product vectors using `index.query()`, returning the top-K most semantically similar products by cosine similarity.

```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create index
client.create_index(name="ecommerce_products", dimension=384,
                    space_type="cosine", precision=Precision.INT8)

index = client.get_index("ecommerce_products")

# Upsert a product vector
index.upsert([{
    "id": "1",
    "vector": [...],   # 384-dim embedding
    "meta": {"name": "Sony WH-1000XM5", "price": 349.99, ...}
}])

# Semantic search
results = index.query(vector=[...], top_k=5)
```

---

## 📂 Project Structure

```
ecommerce-rag-endee/
├── app/
│   ├── main.py            # FastAPI app & endpoints
│   └── rag_pipeline.py    # Embedding, Endee indexing, retrieval, generation
├── data/
│   └── products.json      # 15 sample e-commerce products
├── scripts/
│   └── demo.py            # Demo script to test all endpoints
├── docker-compose.yml     # Runs Endee + FastAPI together
├── Dockerfile             # Container for the FastAPI app
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup & Execution

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Step 1 — Fork & Clone

Per the submission requirements:

1. **Star** the Endee repo: [github.com/endee-io/endee](https://github.com/endee-io/endee)
2. **Fork** it to your GitHub account
3. Clone this project repo:

```bash
git clone https://github.com/<your-username>/ecommerce-rag-endee
cd ecommerce-rag-endee
```

### Step 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env if needed (OpenAI key is optional)
```

### Option A: Run with Docker Compose (Recommended)

Starts both Endee and the FastAPI app with one command:

```bash
docker compose up --build
```

- Endee dashboard: [http://localhost:8080](http://localhost:8080)
- RAG API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Run Locally

**1. Start Endee via Docker:**

```bash
docker run -p 8080:8080 -v endee-data:/data endeeio/endee-server:latest
```

**2. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**3. Start the FastAPI server:**

```bash
uvicorn app.main:app --reload
```

### Step 3 — Ingest Products

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force_reingest": true}'
```

### Step 4 — Query the RAG API

```bash
# RAG question answering
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What headphones are good for travel?", "top_k": 3}'

# Pure semantic search
curl "http://localhost:8000/products/search?q=running+shoes&top_k=3"
```

### Step 5 — Run Full Demo

```bash
python scripts/demo.py
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest products into Endee |
| `POST` | `/query` | RAG: answer a product question |
| `GET` | `/products/search?q=...` | Semantic product search |

Interactive API docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 💡 Example Queries

| Question | What Endee Retrieves |
|----------|----------------------|
| "I need noise-cancelling headphones" | Sony WH-1000XM5, AirPods Pro |
| "Best laptop for students under $1200" | MacBook Air M3 |
| "Good shoes for running" | Adidas Ultraboost 23, Nike Air Max 270 |
| "Something to make coffee quickly" | Nespresso Vertuo Pop |
| "Warm jacket for winter hiking" | The North Face Thermoball Jacket |

---

## 🧠 Technical Highlights

- **Zero hallucination risk** — answers are strictly grounded in retrieved product context
- **INT8 quantization** on Endee for memory-efficient high-speed search
- **Graceful fallback** — works without OpenAI key using rule-based answer generation
- **Async FastAPI** — non-blocking I/O for high throughput
- **Docker-first** — single `docker compose up` to run everything

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)
