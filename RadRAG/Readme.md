# RadRAG — Semantic + Hierarchical

Este projeto implementa um sistema RAG com chunking semântico, hierárquico ou híbrido. A execução depende exclusivamente dos arquivos listados abaixo: `app.py`, `pipeline.py`, `config.yaml` e os módulos dentro de `core/`.

## Arquivos Principais
- `app.py`: Interface de linha de comando (menu) para processar PDFs, buscar e perguntar ao LLM.
- `pipeline.py`: Orquestra toda a pipeline (extração PDF, chunking, indexação no ChromaDB, busca, reranking e geração de resposta).
- `config.yaml`: Configurações do sistema (modelos, estratégia de chunking, ChromaDB, reranker, etc.).
- `core/__init__.py`: Exposição dos componentes do núcleo.
- `core/advanced_chunker.py`: Chunker avançado com estratégias `hierarchical`, `semantic`, `hybrid` ou `semantic_hierarchical`.
- `core/pdf_processor.py`: Extração de texto estruturado de PDFs (backend PyMuPDF/fitz).
- `core/chromadb_manager.py`: Gerenciamento de coleção ChromaDB (persistência, busca semântica/lexical e híbrida).
- `core/llm_generator.py`: Geração de respostas via Ollama (LLM local) a partir do contexto recuperado.
- `core/reranker.py`: Reranking com `CrossEncoder` (Sentence-Transformers) — opcional.

## Instalação
1) (Opcional) Crie um ambiente virtual
```
python -m venv .venv
.venv\Scripts\activate
```
2) Instale as dependências
```
pip install -r requirements.txt
```
3) Baixe o modelo do spaCy PT caso a instalação do pacote do modelo falhe
```
python -m spacy download pt_core_news_sm
```

## Execução
- Execute o menu CLI:
```
python app.py
```
- Configure caminhos e modelos em `config.yaml` conforme necessário.

