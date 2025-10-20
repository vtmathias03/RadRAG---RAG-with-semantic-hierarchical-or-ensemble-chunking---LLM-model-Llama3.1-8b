# RAG OpenAI — Streamlit + LangChain

Este projeto implementa um RAG simples com persistência de vetores usando ChromaDB e embeddings da OpenAI, exposto via interface web em Streamlit.

## Arquivos Principais
- `rag_app_openai.py`: Aplicação Streamlit que gerencia upload de PDFs, construção do vetorstore (Chroma), recuperação com memória de conversa e geração de respostas via OpenAI.
- `rag_data/`: Diretório de dados persistentes
  - `rag_data/pdfs/`: PDFs armazenados com nome normalizado e hash
- `.env`: Arquivo com `OPENAI_API_KEY` (obrigatório)

## Requisitos
Instale as dependências:
```
pip install -r requirements.txt
```
Crie um arquivo `.env` na raiz do projeto com:
```
OPENAI_API_KEY="sua-chave-aqui"
```

## Execução
Inicie a interface web do RAG:
```
streamlit run rag_app_openai.py
```
Após iniciar:
- Use a barra lateral para adicionar PDFs e gerenciar o banco de vetores
- Faça perguntas no chat; as respostas vêm do contexto indexado

## Notas
- O vetorstore é persistente em `rag_data/vector_db`.
- A remoção/reconstrução do banco pode ser feita pela barra lateral.
- O app usa o modelo `gpt-4o-mini` via `langchain-openai` por padrão (ajustável no código).
