# Relatório do Sistema RAG (Semantic + Hybrid)


Inclui o papel de cada arquivo, como os módulos se conectam, quais parâmetros podem ser ajustados e recomendações práticas para a configuração atual (geração de chunks híbrida: semantic + hierarchical).

---

## Visão Geral do Fluxo

1) Extração de texto de PDFs
- Arquivo: `core/pdf_processor.py`
- Extrai o texto e metadados (por página ou via sumário/TOC), normalizando para uso posterior.

2) Criação de chunks (fragmentos) do texto
- Arquivo: `core/advanced_chunker.py`
- Divide o texto em partes (pais/filhos e/ou grupos semânticos) para indexação eficiente.

3) Indexação e busca em ChromaDB
- Arquivo: `core/chromadb_manager.py`
- Persiste os chunks e realiza buscas semânticas, por palavras-chave e híbridas.

4) Orquestração do pipeline e geração de resposta
- Arquivo: `pipeline.py`
- Coordena as etapas; na pergunta do usuário, busca os chunks relevantes e chama o LLM.

5) LLM (modelo de linguagem) local via Ollama
- Arquivo: `core/llm_generator.py`
- Monta o prompt, chama o modelo e retorna a resposta, com regras para evitar alucinações.

6) Interface de linha de comando (CLI)
- Arquivo: `app.py`
- Menu para processar PDF(s), buscar documentos e fazer perguntas ao LLM.

---

## Estrutura dos Arquivos e Papéis

### 1) `app.py`
- O que é: interface CLI (linha de comando) para operar o sistema.
- O que faz:
  - Exibe um menu simples: processar PDF, processar diretório, buscar, perguntar (com LLM), estatísticas e reset do banco.
  - Inicializa o pipeline a partir do `config.yaml`.
  - Configura a consola para UTF-8 (no Windows, força `chcp 65001`) e imprime acentos/emojis corretamente.
- Principais pontos:
  - Chama `CNENPipeline("config.yaml")` para usar as configurações definidas no YAML.
  - Ao fazer pergunta, mede a relevância dos documentos e exibe fontes usadas.

### 2) `pipeline.py`
- O que é: o “cérebro” do sistema; orquestra componentes.
- Componentes internos que ele instancia:
  - `SentenceTransformer` (modelo de embeddings, conforme `embeddings.model` do YAML; ex.: `BAAI/bge-m3`).
  - `AdvancedChunker` (estratégia conforme `advanced_chunking.strategy`).
  - `ChromaDBManager` (vetor store com persistência em `chroma_db/`).
  - `PDFProcessor` (método `pymupdf` para extração de texto).
  - `OllamaGenerator` (cliente do LLM local).
- Funções-chave:
  - `process_document(pdf_path)`: extrai texto do PDF, chama o chunker por seção e indexa os chunks no ChromaDB. Retorna estatísticas.
  - `process_directory(dir)`: processa todos os PDFs do diretório.
  - `search(query, top_k)`: busca chunks relevantes (usa tipo `semantic` ou `hybrid` conforme config).
  - `search_with_context(query, top_k)`: além dos resultados, agrega o texto do chunk “pai” (quando disponível) para dar contexto completo.
  - `ask(query, top_k)`: faz a busca, monta o contexto e aciona o LLM para gerar a resposta; se não houver contexto suficiente, retorna uma mensagem padronizada de ausência de informação.
  - `get_stats()` e `reset()` para estatísticas e limpeza do banco.
- Detalhes técnicos úteis:
  - Fallback de GPU: se `embeddings.device` estiver em `cuda:X` mas não houver CUDA, cai para `cpu` automaticamente, registrando aviso.
  - Limite de VRAM por processo: quando em CUDA, tenta reservar apenas 80% da VRAM (útil para estabilidade em placas com pouca memória).
  - Filtro de relevância: descarta resultados abaixo de `retrieval.min_relevance_score` para evitar contexto fraco e reduzir alucinações.

### 3) `core/pdf_processor.py`
- O que é: extrator de texto de PDFs.
- Modos suportados:
  - `pypdf2`: leve e simples, útil quando PyMuPDF não se adequa.
  - `pymupdf` (em uso): mais robusto para PDFs complexos (TOC, mais fidelidade de layout, funcionou melhor com funções matemáticas).
- Funções-chave:
  - `extract_text(path)`: texto simples juntando páginas.
  - `extract_structured_text(path)`: texto com metadados e seções; usa TOC quando possível, senão quebra por página.
  - `extract_pages_range(path, start, end)`: extrai um intervalo de páginas.
- Observação prática: como em PDFs há hifenização de fim de linha (ex.: “prote-
ção”), considere normalizar (remover “-
”, `
` → espaço, retirar soft hyphen U+00AD) antes do chunking para evitar palavras “cortadas” no início de chunks.

### 4) `core/advanced_chunker.py`
- O que é: criador de chunks (fragmentos) do texto.
- Modelo de dados: `HierarchicalChunk` (id, texto, nível `parent`/`child`, estratégia, parent_id, embedding opcional, metadados, cluster_id, coherence_score).
- Estratégias disponíveis (sem ENSEMBLE):
  - `hierarchical`: cria `parents` por tamanho (palavras) com `overlap`, depois `children` menores dentro de cada `parent`.
  - `semantic`: segmenta em sentenças, calcula embeddings e agrupa por similaridade (clustering), criando chunks coesos por tópico.
  - `hybrid`: combina hierarchical + clustering semântico para `children`.
  - `semantic_hierarchical`: gera hierarchical e semantic na indexação (sem ponderação especial); a busca híbrida combina resultados.
- Parâmetros (defaults no `__init__`; os valores efetivos vêm do `config.yaml`):
  - `parent_chunk_size`, `parent_overlap` (pais): tamanho e sobreposição em palavras (passo efetivo = tamanho − overlap).
  - `child_chunk_size`, `child_overlap` (filhos): idem dentro de cada pai.
  - `similarity_threshold`: agressividade de agrupamento semântico (mais alto → clusters menores e mais coesos).
  - `buffer_size`: inclui sentenças vizinhas aos clusters para mais contexto.
  - `min_chunk_size`, `max_chunk_size`: saneamento de extremos.
- Recomendações para normas técnicas:
  - Pais: 800–1200 palavras; overlap 150–250.
  - Filhos: 300–450 palavras; overlap 60–120.
  - `similarity_threshold`: 0.72–0.80; `buffer_size`: 1–2; `min_chunk_size` ≥ 80; `max_chunk_size` 1200–1800.
- Dica para evitar cortes no meio de sentenças:
  - Usar segmentação por sentenças como unidade de acumulação (e overlap por sentenças), ou normalizar o texto (remover hifenização de fim de linha) antes do chunking.

### 5) `core/chromadb_manager.py`
- O que é: camada de persistência e busca no ChromaDB (vetor store).
- Inicialização:
  - Usa `chromadb.PersistentClient` com diretório `./chroma_db`.
  - A coleção é criada/aberta com função de embedding fornecida pelo pipeline (o pipeline usa o mesmo modelo de embeddings para consultas e documentos, garantindo consistência 1024-dim do `bge-m3`).
  - Em caso de conflito de função de embedding persistida, há lógica de recriação segura da coleção.
- Funções-chave:
  - `add_chunks(chunks)`: insere em batches; inclui metadados úteis (nível, parent_id, cluster_id, coerência, tamanho).
  - `search(query_embedding, n_results)`: busca semântica pura (retorna também `score` calculado a partir da distância conforme a métrica).
  - `hybrid_search(query_text, query_embedding, n_results, alpha)`: combina busca semântica e por palavras-chave.
    - O peso semântico é `alpha` (config `chromadb.search.semantic_weight`), e o textual é o complemento (1 − alpha).
  - `delete_documents`, `update_document`, `get_collection_stats`, `reset_collection`, `get_by_parent`.
- Recomendações:
  - Métrica: `cosine` + embeddings normalizados (`embeddings.normalize: true`) é estável.
  - Ajustar `alpha` entre 0.6–0.8 para consultas “semânticas” mantendo sensibilidade a termos de norma (códigos, siglas).

### 6) `core/llm_generator.py`
- O que é: conector com o LLM (Ollama) para gerar a resposta final.
- Como funciona:
  - Monta um `system_prompt` com regras claras para evitar alucinações (usar apenas o contexto; se faltar, responder exatamente com a mensagem padrão de ausência).
  - Constrói um `user_prompt` com os trechos de contexto (documento + seção + texto).
  - Chama `POST /api/generate` do Ollama, com `temperature` e `num_predict` (derivados de `llm.temperature` e `llm.max_tokens`).
  - Versão streaming também disponível.
- Parâmetros úteis:
  - `llm.temperature`: 0.1–0.3 para respostas técnicas mais determinísticas.
  - `llm.max_tokens`: aumentar se notar respostas cortadas (1000 → 2000/3000, conforme o modelo suporte).
  - Timeout HTTP: pode ser ampliado para respostas longas.

### 7) `config.yaml`
- O que é: arquivo principal de configuração, lido pelo pipeline na inicialização.
- Seções principais:
  - `app`: nome/versão (apenas informativo).
  - `advanced_chunking`: estratégia e parâmetros do chunker (os valores daqui sobrescrevem os defaults do `__init__`).
  - `chromadb`: persistência, coleção, métrica e tipo de busca (`semantic`, `keyword` ou `hybrid`). Em híbrido, `semantic_weight` define o peso semântico (o textual é 1 − esse valor).
  - `llm`: modelo Ollama, `base_url`, `temperature`, `max_tokens`.
  - `embeddings`: modelo de embeddings, device (GPU/CPU), batch size, normalização, FP16 (quando suportado).
  - `retrieval`: `top_k_final` (quantos chunks levar ao LLM), `min_relevance_score` (corte de baixa relevância), `use_parent_context` (traz pai para contexto), etc.
  - `performance` e `logging` (opcionais/operacionais).
- Recomendações de ajuste para o modo atual:
  - `advanced_chunking.strategy`: `semantic_hierarchical` ou `hybrid`.
  - `chromadb.search.type`: `hybrid`, com `semantic_weight` em 0.7–0.8.
  - `retrieval.top_k_final`: 5–8 (mais contexto, mais custo); `min_relevance_score`: 0.5–0.65.
  - `llm.max_tokens`: subir se houver cortes; `temperature`: 0.1–0.25.
  - `embeddings.batch_size`: 8–12 (CPU) ou 16–32 (GPU), conforme VRAM.

---

## Parâmetros ajustáveis (e Intervalos Sugeridos)

- Chunking (YAML: `advanced_chunking`)
  - `strategy`: `semantic_hierarchical` (mistura hierarchical + semantic sem pesos especiais) ou `hybrid`.
  - `parent_chunk_size`/`parent_overlap`: 800–1200 / 150–250.
  - `child_chunk_size`/`child_overlap`: 300–450 / 60–120.
  - `similarity_threshold`: 0.72–0.80.
  - `buffer_size`: 1–2.
  - `min_chunk_size`/`max_chunk_size`: ≥ 80 / 1200–1800.

- Busca (YAML: `chromadb.search`)
  - `type`: `hybrid` (combina semântico + lexical).
  - `semantic_weight`: 0.7–0.8 (o restante é peso de palavras-chave).

- Recuperação (YAML: `retrieval`)
  - `top_k_final`: 5–8.
  - `min_relevance_score`: 0.5–0.65.
  - `use_parent_context`: `true` (traz o pai do chunk para o LLM quando existir).

- LLM (YAML: `llm`)
  - `temperature`: 0.1–0.25.
  - `max_tokens`: 2000–3000 (subir se notar cortes em respostas longas).
  - (Opcional) Aumentar timeout HTTP no gerador para 90–120s se necessário.

- Embeddings (YAML: `embeddings`)
  - `device`: `cuda:X` se houver GPU; caso contrário `cpu` (o pipeline faz fallback).
  - `batch_size`: 8–12 (CPU), 16–32 (GPU).
  - `normalize`: `true` (recomendado com `cosine`).
  - `use_fp16`: `true` em GPUs compatíveis para reduzir VRAM.

---

## Como os Arquivos se Referenciam

- `app.py` → cria `CNENPipeline` (`pipeline.py`) e chama métodos: processar, buscar, perguntar, estatísticas.
- `pipeline.py` → usa `core/advanced_chunker.py` (chunking), `core/chromadb_manager.py` (vetor store), `core/pdf_processor.py` (extração), `core/llm_generator.py` (LLM), e carrega `config.yaml`.
- `core/chromadb_manager.py` → usa a função de embedding fornecida pelo `pipeline` (mesmo modelo de embeddings) para garantir consistência na consulta e indexação.
- `core/llm_generator.py` → não conhece ChromaDB; recebe apenas a lista de chunks do `pipeline` e retorna texto e fontes.


