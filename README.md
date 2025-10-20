# RadRAG---RAG-with-semantic-hierarchical-or-ensemble-chunking---LLM-model-Llama3.1-8b

RadRAG: an advanced RAG developed for a capstone project, whose purpose is to answer questions on radiological protection based on Brazil’s CNEN standards. A benchmarking was conducted across different chunking techniques and comparing RadRAG with freely available commercial LLMs.

RadRAG: RAG avançado, desenvolvido para um Trabalho de Conclusão de Curso, cuja função é responder sobre perguntas de radioproteção com base nas normas da CNEN (Brasil). Foi realizado um benchmarking entre diferentes técnicas de chunking e do RadRAG com LLMs comerciais disponíveis gratuitamente.

### Goal
- RAG system focused on CNEN standards with semantic, hierarchical, hybrid, and ensemble (semantic_hierarchical) chunking strategies, semantic and hybrid search, reranking, and answer generation via a local LLM (Ollama).

### Tools & Technologies
- Language/Runtime
  - Python
- Vector store
  - ChromaDB (`chromadb.PersistentClient`)
- Embeddings & Reranking
  - Sentence-Transformers (default: `BAAI/bge-m3`, 1024 dimensions)
  - PyTorch (`torch`) with optional CUDA support (per-process memory cap when available)
  - CrossEncoder for reranking (default: `BAAI/bge-reranker-v2-m3`)
- NLP utilities (sentence splitting)
  - spaCy (`pt_core_news_sm`) — preferred
  - NLTK — fallback
  - Regex — final fallback
- PDF I/O
  - PyMuPDF (`fitz`) for plain and structured text extraction (with TOC when available)
- Infra/Utilities
  - `scikit-learn` (AgglomerativeClustering; cosine_similarity)
  - `numpy`, `tqdm`, `PyYAML`, `requests`
  - `rank-bm25` (optional) for lexical signal in hybrid search
  - Local LLM via Ollama (HTTP API)

### Methodologies
- Ingestion & Extraction
  - Extraction with PyMuPDF (page text; attempts structure via TOC).
  - Collected metadata: `title`, `author`, `subject`, `creator`, total pages.
- Chunking Strategies (core/advanced_chunker.py)
  - Hierarchical: sliding-window "parent" (overlap) + division into "child" within each parent.
  - Semantic: sentence embeddings + AgglomerativeClustering (distance = 1 − cos) with buffer expansion; intra-cluster coherence scoring.
  - Hybrid: hierarchical parents, children derived by semantic clusters from the parent content.
  - Semantic_hierarchical (ensemble): runs hierarchical and semantic pipelines, merging chunks (equal weight at retrieval time).
  - Main controls: `parent_chunk_size`, `child_chunk_size`, `overlap`, `similarity_threshold`, `buffer_size`, bounds `min_chunk_size`/`max_chunk_size`, `sentence_splitter`.
- Embedding Generation
  - `SentenceTransformer` with configurable batching and normalization; device set via `config.yaml` (CPU/CUDA).
- Indexing (core/chromadb_manager.py)
  - Persistence via `PersistentClient`; collection created/retrieved by name.
  - Per-chunk metadata: `level`, `parent_id`, `cluster_id`, `coherence_score`, `chunk_size`, `strategy`, source/pages/section.
- Retrieval
  - Semantic: embedding query with score derived from distance based on configured function.
  - Hybrid: combine dense (semantic) and lexical (BM25 when available; Chroma textual fallback), weighting `alpha` (semantic) and `1−alpha` (lexical).
  - Optional filters: `where`, `where_document` (Chroma) and `min_relevance_score` (post-filtering).
- Reranking (core/reranker.py)
  - `CrossEncoder` scores (query, text) pairs; replaces `final_score` with reranker score. Device auto-selected (CUDA/CPU).
- Hierarchical Context (pipeline.py)
  - When a child result is returned, fetch its parent via `parent_id` and compose `full_context` (general context + specific detail).
- Answer Generation (core/llm_generator.py)
  - Ollama `generate` API with `system` and `prompt` composed from retrieved chunks; streaming supported.
- Metrics & Observability
  - `get_collection_stats()`: total chunks, unique sources, distribution by strategy/level, average coherence, average chunk size, etc.
  - Logging configurable (level/format/file) via `config.yaml`.

### Configuration (config.yaml)
- `advanced_chunking`: strategy (`hierarchical`, `semantic`, `hybrid`, `semantic_hierarchical`), sizes/overlaps, thresholds, and splitters.
- `chromadb`: persistence directory, collection name, distance function, and hybrid search weights (`semantic_weight`).
- `llm`: Ollama model, `base_url`, temperature, and `max_tokens`.
- `embeddings`: model, `device`, `batch_size`, `normalize`.
- `retrieval`: `top_k_initial`/`top_k_final`, `min_relevance_score`, `use_parent_context`.
- `reranker`: `enabled`, `model`, `device`, `batch_size`, `top_k_rerank`.
  
### High-Level Flow
1) PDF → extraction (text/TOC) → sections + metadata
2) Section → chunking (configured strategy) → chunks + embeddings
3) Chunks → indexing in ChromaDB (persistence + rich metadata)
4) Query → embedding → semantic or hybrid search (alpha)
5) (Optional) Reranking with CrossEncoder
6) (Optional) Context enrichment with parent
7) Answer generation via Ollama

### Quick Start
- Install: `pip install -r requirements.txt`
- Run CLI: `python app.py`
- Configuration file: `config.yaml`

### Environment Requirements
- CPU works; GPU (CUDA) optionally accelerates embeddings and reranking.
- For spaCy PT: `python -m spacy download pt_core_news_sm` (if needed).
- For full hybrid search: install `rank-bm25`.

### Limitations & Notes
- `semantic_hierarchical` does not apply different weights across strategies during retrieval; both contribute equally.
- If `CrossEncoder` is not available, the system proceeds without reranking.
- OLLAMA must be running locally at the configured `base_url`.

## RadRAG em português 

### Objetivo
- Sistema RAG focado nas normas CNEN com estratégias de chunking semântico, hierárquico, híbrido e ensemble (semantic_hierarchical), busca semântica e híbrida, reranking e geração de respostas via LLM local (Ollama).

### Ferramentas & Tecnologias
- Linguagem/Runtime
  - Python
- Vetor store
  - ChromaDB (`chromadb.PersistentClient`) com persistência 
- Embeddings & Reranking
  - Sentence-Transformers (modelo padrão: `BAAI/bge-m3`, 1024 dimensões)
  - PyTorch (`torch`) com suporte opcional a CUDA (limite de memória por processo quando disponível)
  - CrossEncoder para reranking (padrão: `BAAI/bge-reranker-v2-m3`)
- NLP utilitários (sentence splitting)
  - spaCy (`pt_core_news_sm`) — preferencial
  - NLTK — fallback
  - Regex — fallback final
- PDF I/O
  - PyMuPDF (`fitz`) para extração de texto simples e estruturado (com TOC quando disponível)
- Infra/Utilitários
  - `scikit-learn` (AgglomerativeClustering; cosine_similarity)
  - `numpy`, `tqdm`, `PyYAML`, `requests`
  - `rank-bm25` (opcional) para sinal lexical na busca híbrida
  - LLM local via Ollama (HTTP API)

### Metodologias
- Ingestão e Extração
  - Extração com PyMuPDF (texto por página; tentativa de estrutura via TOC).
  - Metadados coletados: `title`, `author`, `subject`, `creator`, páginas totais.
- Estratégias de Chunking (core/advanced_chunker.py)
  - Hierarchical: janela deslizante de "parent" (overlap) + divisão em "child" dentro do parent.
  - Semantic: embeddings de sentenças + AgglomerativeClustering (distância = 1 − cos) com expansão de buffer; cálculo de coerência intra-cluster.
  - Hybrid: parents hierárquicos, children derivados por clusters semânticos do conteúdo do parent.
  - Semantic_hierarchical (ensemble): executa pipelines de hierarchical e semantic, unindo os chunks (peso igual no retrieval).
  - Controles principais: `parent_chunk_size`, `child_chunk_size`, `overlap`, `similarity_threshold`, `buffer_size`, limites `min_chunk_size`/`max_chunk_size`, `sentence_splitter`.
- Geração de Embeddings
  - `SentenceTransformer` com lote e normalização configuráveis; device definido por `config.yaml` (CPU/CUDA).
- Indexação (core/chromadb_manager.py)
  - Persistência via `PersistentClient`; coleção criada/recuperada por nome.
  - Metadados por chunk: `level`, `parent_id`, `cluster_id`, `coherence_score`, `chunk_size`, `strategy`, origem/páginas/section.
- Recuperação
  - Semântica: consulta por embedding com cálculo de score a partir da distância, conforme função de distância.
  - Híbrida: combinação de denso (semântica) e lexical (BM25 quando disponível; fallback textual do Chroma), com ponderação `alpha` (semântica) e `1−alpha` (lexical).
  - Filtros opcionais: `where`, `where_document` (Chroma) e `min_relevance_score` (pós-processamento).
- Reranking (core/reranker.py)
  - `CrossEncoder` pontua pares (query, texto); substitui `final_score` pelo score do reranker. Dispositivo auto-selecionado (CUDA/CPU).
- Contexto Hierárquico (pipeline.py)
  - Quando o resultado é child, busca o parent pelo `parent_id` e compõe `full_context` (contexto geral + detalhe específico).
- Geração de Resposta (core/llm_generator.py)
  - Ollama `generate` API com `system` e `prompt` construídos a partir dos chunks recuperados; suporte a resposta streaming.
- Métricas & Observabilidade
  - `get_collection_stats()`: total de chunks, fontes únicas, distribuição por estratégia/nível, coerência média, tamanho médio dos chunks, etc.
  - Logging configurável (nível/formato/arquivo) via `config.yaml`.

### Configuração (config.yaml)
- `advanced_chunking`: estratégia (`hierarchical`, `semantic`, `hybrid`, `semantic_hierarchical`), tamanhos/overlaps, thresholds e splitters.
- `chromadb`: diretório de persistência, nome da coleção, função de distância e pesos da busca híbrida (`semantic_weight`).
- `llm`: modelo Ollama, `base_url`, temperatura e `max_tokens`.
- `embeddings`: modelo, `device`, `batch_size`, `normalize`.
- `retrieval`: `top_k_initial`/`top_k_final`, `min_relevance_score`, `use_parent_context`.
- `reranker`: `enabled`, `model`, `device`, `batch_size`, `top_k_rerank`.
- `performance` e `logging`.

### Fluxo de Alto Nível
1) PDF → extração (texto/TOC) → seções + metadados
2) Seção → chunking (estratégia configurada) → chunks + embeddings
3) Chunks → indexação no ChromaDB (persistência + metadados ricos)
4) Query → embedding → busca semântica ou híbrida (alpha)
5) (Opcional) Reranking com CrossEncoder
6) (Opcional) Enriquecimento de contexto com parent
7) Geração de resposta via Ollama

### Execução Rápida
- Instalação: `pip install -r requirements.txt`
- Execução CLI: `python app.py`
- Arquivo de configuração: `config.yaml`

### Requisitos de Ambiente
- CPU funciona; GPU (CUDA) opcional acelera embeddings e reranking.
- Para spaCy PT: `python -m spacy download pt_core_news_sm` (se necessário).
- Para busca híbrida completa: instalar `rank-bm25`.

### Limitações e Notas
- `semantic_hierarchical` não aplica pesos diferentes entre estratégias na recuperação; ambas contribuem igualmente.
- Se `CrossEncoder` não estiver disponível, o sistema segue sem reranking.
- OLLAMA deve estar rodando localmente no `base_url` especificado.

