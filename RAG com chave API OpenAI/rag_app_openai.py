# ====================================================================
# SISTEMA RAG COM PERSISTÊNCIA DE VETORES
# ====================================================================
# Sistema com banco de vetores persistente e gerenciamento de documentos
# ====================================================================

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# ====================================================================
# CONFIGURAÇÃO DE DIRETÓRIOS PERSISTENTES
# ====================================================================
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "rag_data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
PDF_STORAGE_DIR = DATA_DIR / "pdfs"
METADATA_FILE = DATA_DIR / "documents_metadata.json"

CHROMA_COLLECTION_NAME = "rag_documents"
CHROMA_ARGS = {
    "collection_name": CHROMA_COLLECTION_NAME,
    "persist_directory": str(VECTOR_DB_DIR),
    "collection_metadata": {"hnsw:space": "cosine"},
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COSINE_DUPLICATE_THRESHOLD = 0.96

for directory in (DATA_DIR, VECTOR_DB_DIR, PDF_STORAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ====================================================================
# FUNÇÕES UTILITÁRIAS DE METADADOS
# ====================================================================

def load_metadata() -> dict:
    """Carrega metadados dos documentos salvos."""
    if not METADATA_FILE.exists():
        return {"documents": {}, "last_updated": None}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"documents": {}, "last_updated": None}

    metadata.setdefault("documents", {})
    return metadata


def save_metadata(metadata: dict) -> None:
    """Salva metadados dos documentos."""
    metadata.setdefault("documents", {})
    metadata["last_updated"] = datetime.utcnow().isoformat()

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def get_file_hash(file_content: bytes) -> str:
    """Gera hash único para identificar o arquivo."""
    import hashlib

    return hashlib.md5(file_content).hexdigest()


def sanitize_filename(name: str) -> str:
    """Normaliza o nome do arquivo para armazenamento."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return sanitized.strip("_") or "documento"


def store_pdf(file_content: bytes, original_name: str, file_hash: str) -> Path:
    """Persiste o PDF original para permitir reconstrução futura."""
    suffix = Path(original_name).suffix or ".pdf"
    stem = sanitize_filename(Path(original_name).stem)
    stored_name = f"{file_hash}_{stem}{suffix.lower()}"
    stored_path = PDF_STORAGE_DIR / stored_name

    with open(stored_path, "wb") as pdf_file:
        pdf_file.write(file_content)

    return stored_path


@st.cache_resource(show_spinner=False)
def initialize_embeddings() -> OpenAIEmbeddings:
    """Inicializa os embeddings da OpenAI."""
    return OpenAIEmbeddings()


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calcula a similaridade de cosseno entre dois vetores."""
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def filter_similar_chunks(chunks: List, embeddings: OpenAIEmbeddings, threshold: float) -> List:
    """Remove chunks redundantes com alta similaridade de cosseno."""
    if not chunks:
        return []

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    kept_chunks: List = []
    kept_vectors: List[np.ndarray] = []

    for chunk, vector in zip(chunks, vectors):
        np_vector = np.array(vector, dtype=np.float32)

        if not kept_vectors:
            kept_chunks.append(chunk)
            kept_vectors.append(np_vector)
            continue

        similarities = [cosine_similarity(np_vector, stored) for stored in kept_vectors]
        if not similarities or max(similarities) < threshold:
            kept_chunks.append(chunk)
            kept_vectors.append(np_vector)

    return kept_chunks


def build_chunks_from_pdf(
    pdf_path: Path,
    display_name: str,
    file_hash: str,
    embeddings: OpenAIEmbeddings,
    threshold: float = COSINE_DUPLICATE_THRESHOLD,
) -> Tuple[List, int]:
    """Carrega um PDF, gera chunks e enriquece metadados."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    page_count = len(pages)

    splitter = create_text_splitter()
    chunks = splitter.split_documents(pages)
    filtered_chunks = filter_similar_chunks(chunks, embeddings, threshold)

    for chunk in filtered_chunks:
        chunk.metadata["source"] = str(pdf_path)
        chunk.metadata["file_name"] = display_name
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["page"] = chunk.metadata.get("page", 0)

    return filtered_chunks, page_count


def reset_chat_state() -> None:
    """Limpa histórico de mensagens e memória."""
    st.session_state.messages = []
    if st.session_state.get("memory") is not None:
        st.session_state.memory.clear()


def add_documents_to_vectorstore(files: List) -> None:
    """Adiciona novos documentos ao banco de vetores."""
    if not files:
        return

    metadata = st.session_state.documents_metadata
    metadata.setdefault("documents", {})

    embeddings = initialize_embeddings()

    new_chunks: List = []
    processed_documents: List[str] = []
    total_pages = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for index, file in enumerate(files, start=1):
            file_content = file.read()
            file_hash = get_file_hash(file_content)

            if file_hash in metadata["documents"]:
                st.warning(f"{file.name} já existe no banco de vetores.")
                file.seek(0)
                progress_bar.progress(index / len(files))
                continue

            status_text.text(f"Processando: {file.name}")
            stored_path = store_pdf(file_content, file.name, file_hash)

            chunks, page_count = build_chunks_from_pdf(
                stored_path,
                file.name,
                file_hash,
                embeddings,
            )

            if not chunks:
                st.warning(f"Nenhum conteúdo válido encontrado em {file.name}.")
                continue

            metadata["documents"][file_hash] = {
                "hash": file_hash,
                "name": file.name,
                "path": str(stored_path.relative_to(BASE_DIR)),
                "pages": page_count,
                "chunks": len(chunks),
                "added_at": datetime.utcnow().isoformat(),
            }

            new_chunks.extend(chunks)
            processed_documents.append(file.name)
            total_pages += page_count

            progress_bar.progress(index / len(files))

        if not new_chunks:
            status_text.empty()
            progress_bar.empty()
            st.info("Nenhum novo documento foi adicionado.")
            return

        status_text.text("Adicionando ao banco de vetores...")

        if any(VECTOR_DB_DIR.iterdir()):
            vectorstore = Chroma(embedding_function=embeddings, **CHROMA_ARGS)
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = Chroma.from_documents(
                documents=new_chunks,
                embedding=embeddings,
                **CHROMA_ARGS,
            )

        vectorstore.persist()
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_metadata = metadata
        save_metadata(metadata)
        create_conversation_chain()
        reset_chat_state()

        status_text.empty()
        progress_bar.empty()

        st.success(
            "\n".join(
                [
                    "Documentos adicionados com sucesso!",
                    f"• Arquivos processados: {len(processed_documents)}",
                    f"• Páginas analisadas: {total_pages}",
                    f"• Chunks gerados: {len(new_chunks)}",
                ]
            )
        )

    except Exception as exc:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Erro ao processar documentos: {exc}")


def remove_document_from_vectorstore(doc_hash: str, doc_name: str) -> bool:
    """Remove um documento do sistema e reconstrói o banco."""
    metadata = st.session_state.documents_metadata
    document = metadata.get("documents", {}).get(doc_hash)

    if not document:
        st.warning("Documento não encontrado nos metadados.")
        return False

    stored_path = BASE_DIR / document.get("path", "")
    if stored_path.exists():
        stored_path.unlink(missing_ok=True)

    del metadata["documents"][doc_hash]
    save_metadata(metadata)
    st.session_state.documents_metadata = metadata

    rebuild_vectorstore()
    st.success(f"{doc_name} removido com sucesso.")
    return True


def clear_vectorstore() -> None:
    """Remove completamente o banco de vetores e PDFs armazenados."""
    try:
        if VECTOR_DB_DIR.exists():
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
        if PDF_STORAGE_DIR.exists():
            shutil.rmtree(PDF_STORAGE_DIR, ignore_errors=True)

        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

        metadata = {"documents": {}, "last_updated": None}
        save_metadata(metadata)

        st.session_state.documents_metadata = metadata
        st.session_state.vectorstore = None
        st.session_state.chain = None
        st.session_state.memory = None
        reset_chat_state()

        st.success("Banco de vetores limpo com sucesso.")
    except Exception as exc:
        st.error(f"Erro ao limpar banco de vetores: {exc}")


def rebuild_vectorstore() -> None:
    """Reconstrói o banco de vetores a partir dos PDFs persistidos."""
    metadata = st.session_state.documents_metadata
    documents = metadata.get("documents", {})

    if not documents:
        clear_vectorstore()
        st.info("Nenhum documento disponível para reconstrução.")
        return

    try:
        with st.spinner("Reconstruindo banco de vetores..."):
            if VECTOR_DB_DIR.exists():
                shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

            embeddings = initialize_embeddings()
            rebuilt_chunks: List = []

            for doc_hash, doc_info in documents.items():
                pdf_path = BASE_DIR / doc_info.get("path", "")
                if not pdf_path.exists():
                    st.warning(f"Arquivo ausente: {doc_info.get('name', doc_hash)}")
                    continue

                chunks, page_count = build_chunks_from_pdf(
                    pdf_path,
                    doc_info.get("name", pdf_path.name),
                    doc_hash,
                    embeddings,
                )

                doc_info["pages"] = page_count
                doc_info["chunks"] = len(chunks)
                rebuilt_chunks.extend(chunks)

            if not rebuilt_chunks:
                st.warning("Nenhum chunk disponível para reconstruir o banco.")
                st.session_state.vectorstore = None
            else:
                vectorstore = Chroma.from_documents(
                    documents=rebuilt_chunks,
                    embedding=embeddings,
                    **CHROMA_ARGS,
                )
                vectorstore.persist()
                st.session_state.vectorstore = vectorstore

            save_metadata(metadata)
            st.session_state.documents_metadata = metadata
            create_conversation_chain()
            reset_chat_state()

    except Exception as exc:
        st.error(f"Erro ao reconstruir banco de vetores: {exc}")


def load_existing_vectorstore() -> None:
    """Carrega o banco de vetores persistido."""
    if not VECTOR_DB_DIR.exists() or not any(VECTOR_DB_DIR.iterdir()):
        st.warning("Nenhum banco de vetores encontrado.")
        return

    try:
        with st.spinner("Carregando banco de vetores..."):
            embeddings = initialize_embeddings()
            vectorstore = Chroma(embedding_function=embeddings, **CHROMA_ARGS)
            st.session_state.vectorstore = vectorstore
            create_conversation_chain()
            reset_chat_state()
        st.success("Banco de vetores carregado com sucesso.")
    except Exception as exc:
        st.error(f"Erro ao carregar banco de vetores: {exc}")


# ====================================================================
# CONFIGURAÇÃO DA PÁGINA
# ====================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY or OPENAI_API_KEY == "sua-api-key-aqui":
    st.error(
        """\
⚠️ **ATENÇÃO: A API Key da OpenAI não foi encontrada!**

Para configurar:
1. Crie um arquivo `.env` na mesma pasta deste script
2. Adicione: `OPENAI_API_KEY="sua-chave-real-aqui"`
3. Salve e execute novamente

**Obtenha sua API Key em:** https://platform.openai.com/api-keys
"""
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(
    page_title="Chat para Normas de Radioproteção",
    page_icon="📚",
    layout="wide",
)

st.title("Chat para Normas de Radioproteção")
st.markdown("*Sistema com memória permanente de documentos*")
st.markdown("---")


# ====================================================================
# INICIALIZAÇÃO DO SESSION STATE
# ====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "memory" not in st.session_state:
    st.session_state.memory = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_metadata" not in st.session_state:
    st.session_state.documents_metadata = load_metadata()


def create_conversation_chain() -> None:
    """Cria ou atualiza a cadeia de conversação."""
    if not st.session_state.vectorstore:
        st.session_state.chain = None
        return

    custom_template = """Você é um assistente virtual prestativo.
Use os trechos de contexto recuperados para responder à pergunta.
Se não souber a resposta, informe que a informação não está disponível.
Responda em português e seja conciso, a menos que o usuário peça mais detalhes.
Forneça as referências bibliográficas, sempre que disponível (exemplo: artigo, parágrafo, inciso)

Contexto:
{context}

Histórico da conversa:
{chat_history}

Pergunta:
{question}

Resposta:"""

    prompt = PromptTemplate(
        template=custom_template,
        input_variables=["context", "chat_history", "question"],
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1500,
    )

    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False,
    )


# ====================================================================
# BARRA LATERAL - GERENCIAMENTO DE DOCUMENTOS
# ====================================================================
with st.sidebar:
    st.header("📁 Gerenciamento de Documentos")
    st.success("API Key configurada!")

    metadata = st.session_state.documents_metadata
    documents = metadata.get("documents", {})
    num_docs = len(documents)

    if num_docs > 0:
        last_updated = metadata.get("last_updated")
        formatted_timestamp = last_updated[:19] if last_updated else "N/A"
        st.info(
            f"""\
Status do banco de dados:
• Documentos salvos: **{num_docs}**
• Última atualização: {formatted_timestamp}
"""
        )
    else:
        st.warning("Nenhum documento localizado.")

    st.markdown("---")

    tab_add, tab_manage, tab_system = st.tabs(["➕ Adicionar", "📂 Gerenciar", "⚙️ Sistema"])

    with tab_add:
        st.subheader("Adicionar novos documentos")
        uploaded_files = st.file_uploader(
            "Escolha arquivos PDF",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files and st.button("Adicionar ao banco", use_container_width=True):
            add_documents_to_vectorstore(uploaded_files)

    with tab_manage:
        st.subheader("Documentos salvos")

        if num_docs == 0:
            st.info("Nenhum documento salvo até o momento.")
        else:
            for doc_hash, doc_info in documents.items():
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    st.text(doc_info.get("name", doc_hash))
                    st.caption(
                        f"Páginas: {doc_info.get('pages', 'N/A')} | Chunks: {doc_info.get('chunks', 'N/A')}"
                    )
                with col_right:
                    if st.button(
                        "Remover",
                        key=f"remove_{doc_hash}",
                        help=f"Remover {doc_info.get('name', doc_hash)}",
                    ):
                        if remove_document_from_vectorstore(doc_hash, doc_info.get("name", doc_hash)):
                            st.rerun()

            st.markdown("---")
            if st.button("Limpar banco de vetores", use_container_width=True):
                if st.checkbox("Confirmar exclusão de todos os documentos"):
                    clear_vectorstore()
                    st.rerun()

    with tab_system:
        st.subheader("Configurações do sistema")

        if st.button("Carregar banco de vetores", use_container_width=True):
            load_existing_vectorstore()

        if st.button("Reconstruir banco de vetores", use_container_width=True):
            rebuild_vectorstore()

        if st.button("Limpar histórico do chat", use_container_width=True):
            reset_chat_state()
            st.success("Histórico de chat limpo!")
            st.rerun()

        st.markdown("---")
        st.caption("Modelo: GPT-4o-mini")
        st.caption("Vector store: ChromaDB com similaridade por cosseno")
        st.caption("Stack: LangChain + OpenAI")


# ====================================================================
# INICIALIZAÇÃO AUTOMÁTICA
# ====================================================================
if (
    st.session_state.vectorstore is None
    and VECTOR_DB_DIR.exists()
    and any(VECTOR_DB_DIR.iterdir())
):
    load_existing_vectorstore()


# ====================================================================
# ÁREA DE CHAT PRINCIPAL
# ====================================================================
metadata = st.session_state.documents_metadata
documents = metadata.get("documents", {})
num_docs = len(documents)

if num_docs > 0:
    with st.expander(f"{num_docs} documento(s) disponível(eis)", expanded=False):
        for doc_info in documents.values():
            st.markdown(
                f"**{doc_info.get('name', 'Documento')}** — Páginas: {doc_info.get('pages', 'N/A')} | "
                f"Chunks: {doc_info.get('chunks', 'N/A')}"
            )

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Fontes consultadas"):
                    for index, source in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**Fonte {index}:**\n"
                            f"• Arquivo: `{source.get('file', 'N/A')}`\n"
                            f"• Página: {source.get('page', 'N/A')}\n"
                            f"• Trecho: *{source.get('content', '')[:200]}...*"
                        )


# ====================================================================
# INPUT DE PERGUNTAS
# ====================================================================
if st.session_state.chain and num_docs > 0:
    user_question = st.chat_input("Digite sua pergunta sobre os documentos...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analisando documentos..."):
                try:
                    response = st.session_state.chain({"question": user_question})
                    answer = response.get("answer", "Não foi possível gerar uma resposta.")

                    sources = []
                    for document in response.get("source_documents", [])[:3]:
                        source_path = document.metadata.get("source")
                        source_name = document.metadata.get("file_name")
                        if not source_name and source_path:
                            source_name = Path(source_path).name
                        sources.append(
                            {
                                "file": source_name or "Fonte desconhecida",
                                "page": document.metadata.get("page", 0) + 1,
                                "content": document.page_content,
                            }
                        )

                    st.markdown(answer)

                    if sources:
                        with st.expander("Fontes consultadas"):
                            for index, source in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Fonte {index}:**\n"
                                    f"• Arquivo: `{source['file']}`\n"
                                    f"• Página: {source['page']}\n"
                                    f"• Trecho: *{source['content'][:200]}...*"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    st.error(f"Erro ao gerar resposta: {exc}")
else:
    placeholder_col_left, placeholder_col_center, placeholder_col_right = st.columns([1, 2, 1])
    with placeholder_col_center:
        if num_docs == 0:
            st.info(
                """\
Bem-vindo ao chat com PDFs persistentes!

• Adicione PDFs na barra lateral
• Aguarde o processamento
• Faça perguntas a qualquer momento
"""
            )
        else:
            st.warning("Carregue o banco de vetores na barra lateral para iniciar o chat.")


# ====================================================================
# RODAPÉ
# ====================================================================
st.markdown("---")
st.markdown(
    """\
<div style='text-align: center; color: #666;'>
    <p>GPT-4o-mini | ChromaDB (similaridade por cosseno)</p>
    <p>Documentos salvos permanentemente</p>
</div>
""",
    unsafe_allow_html=True,
)