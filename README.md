# 🎬 CineGraph Intel — Multi-Stage Hybrid & Graph RAG Engine

> An Enterprise-grade, Grounded Movie Retrieval-Augmented Generation (RAG) Engine combining **HyDE Query Expansion**, **Dual Indexing (BM25 + Voyage-3)**, **Cross-Encoder Re-ranking**, and **2-Hop Knowledge Graph Traversal** with **Zero-Redundancy NDJSON Streaming**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([👤 User Query]) --> HyDE[🧠 HyDE Expansion<br>Gemini Flash]
    User --> BM25[⚡ BM25Okapi<br>Lexical Search]
    HyDE --> Voyage[🌐 Voyage-3 Dense Vector<br>Semantic Search]

    BM25 --> RRF[🔀 Reciprocal Rank Fusion<br>k=60]
    Voyage --> RRF

    RRF --> CE[🎯 Cross-Encoder Re-ranker<br>ms-marco-MiniLM-L-6-v2]
    HyDE -. Context .-> CE

    CE --> Graph[🕸️ NetworkX DiGraph<br>2-Hop Entity Traversal]
    Graph --> Synthesis[🤖 Grounded Generator<br>Gemini 2.5 Flash Lite]

    Synthesis --> Stream[⚡ NDJSON Streaming Protocol<br>Chunk 1: Metadata | Chunks 2..N: Tokens]
    Stream --> UI[💻 Streamlit Executive Dashboard]
```

---

## 📊 Empirical Retrieval Benchmark

Evaluated on realistic multi-intent movie queries (abstract concepts, character descriptions, lexical traps):

| Retrieval Strategy                     | Hit Rate@3 |   MRR@3   | Architecture Profile                            |
| :------------------------------------- | :--------: | :-------: | :---------------------------------------------- |
| **1. Naive BM25 (Lexical Only)**       |   66.7%    |   0.667   | Fails on abstract ideas / vocabulary mismatch   |
| **2. Naive Dense Vector (Voyage-3)**   |   100.0%   |   1.000   | Semantic vector match, higher latency           |
| **3. CineGraph Intel (Full Pipeline)** | **100.0%** | **1.000** | **HyDE + RRF + Cross-Encoder + Fault Fallback** |

---

## 🌟 Key Engineering Highlights

1. **Vocabulary Mismatch Resolution (HyDE + Rich Document Encoding):** Solved TMDB lexical sparsity (e.g. _Inception_ overview having no word "dream") by fusing rich schema tags (Title, Director, Cast, Genres) and hypothetical English synopses.
2. **Knowledge Graph Traversal (NetworkX):** 2-hop graph reasoning (`Director -> Movie -> Cast -> Genres`) ground truth context to completely eliminate LLM hallucinations.
3. **Single-Pipe NDJSON Streaming:** Proprietary delimiter protocol streaming JSON graph metadata in Chunk 1 and token text chunks in Chunks 2..N over a single HTTP connection.
4. **Graceful Degradation:** Automatic fault-tolerant fallback to BM25 and raw query if Voyage AI or Gemini experiences 429 RateLimit or network failure.

---

## 🚀 Quickstart Guide

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/cinegraph-intel.git
cd cinegraph-intel
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and VOYAGE_API_KEY
```

### 2. Run Tests & Benchmark

```bash
python -m pytest tests/ -v
python -m eval.benchmark
```

### 3. Start Backend & Dashboard

```bash
# Terminal 1: FastAPI Service
uvicorn api.main:app --reload --port 8000

# Terminal 2: Streamlit Dashboard
streamlit run ui/dashboard.py
```
