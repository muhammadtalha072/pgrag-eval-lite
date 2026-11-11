# pgrag-eval-lite: Local-First PostgreSQL (pgvector + FTS) Hybrid RRF & Table Chunk Benchmark

Reproducible, -API-cost benchmark and pre-index chunk health scanner for testing PostgreSQL 17 (pgvector HNSW + native tsvector GIN + Reciprocal Rank Fusion) against broken Markdown/PDF tables, exact SKUs, and semantic paraphrases.

- **Full Benchmark Write-Up (Dev.to):** https://dev.to/mtalha5/what-broke-when-i-tested-pgvector-vs-postgres-full-text-search-vs-hybrid-rrf-on-30-tricky-rag-4f8l
- **Download pgRAG-Inspector Pro (4-Page FastAPI + HTMX Visual Studio) or Free Lite CLI:** https://mtalha5.gumroad.com/l/pgrag-inspector

## Benchmark Summary (30 Enterprise Chunks / 15 Golden Queries)

| Retrieval Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@10 | NDCG@5 | Mean Latency | API Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Pure Dense (pgvector <=>) | 73.3% | 86.7% | 100.0% | 0.819 | 0.863 | 1.23 ms | .00 |
| 2. Pure Sparse (ts_rank_cd) | 66.7% | 66.7% | 66.7% | 0.667 | 0.667 | 0.74 ms | .00 |
| 3. Hybrid RRF (k=60, id ASC) | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 1.99 ms | .00 |

## Quickstart
Run pip install -r requirements.txt and python benchmark_validation.py.
