"""
Gerenciador ChromaDB otimizado para RAG
Suporta busca semântica, por palavras‑chave e busca híbrida
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
import numpy as np

# BM25 (opcional)
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False

import re
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Gerencia coleções ChromaDB com persistência e otimizações"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "cnen_normas",
        distance_function: str = "cosine",
        embedding_dimension: int = 384,
        embedding_function=None,
    ):
        """
        Args:
            persist_directory: Diretório para persistência
            collection_name: Nome da coleção
            distance_function: Função de distância (cosine, l2, ip)
            embedding_dimension: Dimensão dos embeddings
            embedding_function: Função para gerar embeddings (opcional)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_function = distance_function
        self.embedding_dimension = embedding_dimension
        self.embedding_function = embedding_function

        # Criar diretório se não existir
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Inicializar cliente ChromaDB com persistência
        logger.info(f"Inicializando ChromaDB em {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Criar ou obter coleção
        self._initialize_collection()

    def _initialize_collection(self):
        """Inicializa coleção ChromaDB"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(
                f"Coleção '{self.collection_name}' carregada: {self.collection.count()} documentos"
            )
        except Exception as e:
            logger.warning(f"Falha ao carregar coleção '{self.collection_name}': {e}")
            metadata_config = {"hnsw:space": self.distance_function}
            try:
                self.client.delete_collection(self.collection_name)
            except Exception as delete_err:
                logger.warning(f"Não foi possível deletar coleção existente: {delete_err}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata_config,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Coleção '{self.collection_name}' criada")

    def add_chunks(
        self, chunks: List["HierarchicalChunk"], batch_size: int = 100
    ) -> int:
        """
        Adiciona chunks ao ChromaDB

        Args:
            chunks: Lista de HierarchicalChunk objects
            batch_size: Tamanho do batch para inserção

        Returns:
            Número de chunks adicionados
        """
        logger.info(f"Adicionando {len(chunks)} chunks ao ChromaDB")
        total_added = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids: List[str] = []
            documents: List[str] = []
            embeddings: List[Optional[List[float]]] = []
            metadatas: List[Dict] = []

            for chunk in batch:
                metadata = {
                    **chunk.metadata,
                    "level": chunk.level,
                    "parent_id": chunk.parent_id if chunk.parent_id else "",
                    "cluster_id": int(chunk.cluster_id)
                    if chunk.cluster_id is not None
                    else -1,
                    "coherence_score": float(chunk.coherence_score)
                    if chunk.coherence_score is not None
                    else 0.0,
                    "chunk_size": len((chunk.text or "").split()),
                }
                ids.append(chunk.chunk_id)
                documents.append(chunk.text)
                if getattr(chunk, "embedding", None) is not None:
                    if isinstance(chunk.embedding, np.ndarray):
                        embeddings.append(chunk.embedding.tolist())
                    else:
                        embeddings.append(chunk.embedding)
                else:
                    embeddings.append(None)
                metadatas.append(metadata)

            try:
                if any(e is not None for e in embeddings):
                    valid_embeddings = [e for e in embeddings if e is not None]
                    if len(valid_embeddings) == len(embeddings):
                        self.collection.add(
                            ids=ids,
                            documents=documents,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                    else:
                        # Embeddings parciais, adicionar sem embeddings
                        self.collection.add(
                            ids=ids, documents=documents, metadatas=metadatas
                        )
                else:
                    self.collection.add(
                        ids=ids, documents=documents, metadatas=metadatas
                    )
                total_added += len(batch)
            except Exception as e:
                logger.error(f"Erro ao adicionar batch: {e}")
                raise

        logger.info(f"Total de chunks adicionados: {total_added}")
        logger.info(
            f"Total de documentos na coleção: {self.collection.count()}"
        )
        return total_added

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        include_distances: bool = True,
    ) -> Dict:
        """Busca semântica no ChromaDB"""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        include_list = (
            ["documents", "metadatas", "distances"]
            if include_distances
            else ["documents", "metadatas"]
        )
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include_list,
        )

        formatted_results: List[Dict] = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            if include_distances:
                distance = results["distances"][0][i]
                if self.distance_function == "cosine":
                    score = 1 - distance
                elif self.distance_function == "l2":
                    score = 1 / (1 + distance)
                else:  # inner product
                    score = distance
                result["score"] = float(score)
                result["distance"] = float(distance)
            formatted_results.append(result)

        return {"results": formatted_results, "total": len(formatted_results)}

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        n_results: int = 10,
        alpha: float = 0.7,
        oversample_factor: int = 2,
    ) -> List[Dict]:
        """
        Busca híbrida: combina denso (semântica) e lexical (BM25 quando disponível).
        """
        # 1) Semântica (oversample)
        semantic = self.search(
            query_embedding=query_embedding,
            n_results=n_results * max(1, oversample_factor),
        )

        # 2) Lexical via BM25 (se disponível) ou fallback para query_texts
        keyword_candidates: List[tuple] = []  # (id, score, text, meta)
        if BM25Okapi is not None:
            try:
                data = self.collection.get(include=["documents", "metadatas"])
                ids = data.get("ids", [])
                docs = data.get("documents", [])
                metas = data.get("metadatas", []) or [{} for _ in ids]

                def _normalize(txt: str) -> str:
                    try:
                        txt = (txt or "").lower()
                        txt = unicodedata.normalize("NFKD", txt)
                        txt = "".join(
                            ch for ch in txt if not unicodedata.combining(ch)
                        )
                        return re.sub(r"\W+", " ", txt).strip()
                    except Exception:
                        return txt or ""

                corpus_tokens = [(_normalize(t)).split() for t in docs]
                bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
                q_tokens = (_normalize(query_text)).split()
                scores = bm25.get_scores(q_tokens) if (bm25 and q_tokens) else []
                topk = min(len(scores), n_results * max(1, oversample_factor))
                if topk > 0:
                    idxs = np.argsort(scores)[-topk:][::-1]
                    for i in idxs:
                        keyword_candidates.append((ids[i], float(scores[i]), docs[i], metas[i]))
                logger.info(f"Hybrid search: BM25 ativo (rank_bm25) - {len(keyword_candidates)} candidatos.")
            except Exception as e:
                logger.warning(f"BM25 falhou, usando fallback textual: {e}")

        if not keyword_candidates:
            # Log explícito do status do BM25
            if BM25Okapi is not None:
                if keyword_candidates:
                    logger.info(f"Hybrid search: BM25 ativo (rank_bm25) - {len(keyword_candidates)} candidatos.")
                else:
                    logger.info("Hybrid search: BM25 instalado, sem candidatos; usando fallback textual.")
            else:
                logger.info("Hybrid search: BM25 não instalado; usando fallback textual do Chroma.")
            try:
                kr = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results * max(1, oversample_factor),
                    include=["documents", "metadatas", "distances"],
                )
                for i in range(len(kr["ids"][0])):
                    kid = kr["ids"][0][i]
                    kdoc = kr["documents"][0][i]
                    kmeta = kr["metadatas"][0][i]
                    kscore = float(1 - kr["distances"][0][i])
                    keyword_candidates.append((kid, kscore, kdoc, kmeta))
            except Exception as e:
                logger.warning(f"Fallback textual falhou: {e}")

        # 3) Combinar resultados
        combined: Dict[str, Dict] = {}
        for result in semantic.get("results", []):
            doc_id = result["id"]
            combined[doc_id] = {
                **result,
                "semantic_score": result.get("score", 0) * alpha,
                "keyword_score": 0.0,
            }

        if keyword_candidates:
            ks = [s for _, s, _, _ in keyword_candidates]
            kmin, kmax = min(ks), max(ks)
            denom = (kmax - kmin) + 1e-9
            for kid, kscore, kdoc, kmeta in keyword_candidates:
                k_norm = (kscore - kmin) / denom
                k_weighted = k_norm * (1 - alpha)
                if kid in combined:
                    combined[kid]["keyword_score"] = k_weighted
                    combined[kid]["final_score"] = (
                        combined[kid]["semantic_score"] + k_weighted
                    )
                else:
                    combined[kid] = {
                        "id": kid,
                        "text": kdoc,
                        "metadata": kmeta,
                        "semantic_score": 0.0,
                        "keyword_score": k_weighted,
                        "final_score": k_weighted,
                    }

        results = sorted(
            combined.values(),
            key=lambda x: x.get("final_score", x.get("semantic_score", 0.0)),
            reverse=True,
        )
        return results[:n_results]

    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict] = None,
    ):
        """Atualiza documento existente"""
        update_kwargs = {"ids": [document_id]}
        if document is not None:
            update_kwargs["documents"] = [document]
        if embedding is not None:
            update_kwargs["embeddings"] = [embedding]
        if metadata is not None:
            update_kwargs["metadatas"] = [metadata]
        self.collection.update(**update_kwargs)
        logger.debug(f"Documento {document_id} atualizado")

    def delete_documents(
        self, ids: Optional[List[str]] = None, where: Optional[Dict] = None
    ) -> int:
        """Deleta documentos por IDs ou filtro"""
        count_before = self.collection.count()
        if ids:
            self.collection.delete(ids=ids)
        elif where:
            self.collection.delete(where=where)
        else:
            logger.warning("Nenhum critério de deleção fornecido")
            return 0
        count_after = self.collection.count()
        deleted = count_before - count_after
        logger.info(f"{deleted} documentos deletados")
        return deleted

    def get_collection_stats(self) -> Dict:
        """Retorna estatísticas da coleção"""
        count = self.collection.count()
        if count > 0:
            all_data = self.collection.get(include=["metadatas"])
            coherence_scores = [m.get("coherence_score", 0) for m in all_data["metadatas"]]
            chunk_sizes = [m.get("chunk_size", 0) for m in all_data["metadatas"]]
            unique_sources = sorted({m.get("source", "") for m in all_data["metadatas"] if m.get("source", "")})
            strategy_counts: Dict[str, int] = {}
            level_counts: Dict[str, int] = {}
            for m in all_data["metadatas"]:
                strategy = m.get("strategy", "unknown")
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                level = m.get("level", "unknown")
                level_counts[level] = level_counts.get(level, 0) + 1
            stats = {
                "total_chunks": count,
                "total_pdfs": len(unique_sources),
                "unique_sources": unique_sources,
                "chunks_by_strategy": strategy_counts,
                "chunks_by_level": level_counts,
                "avg_coherence_score": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
                "avg_chunk_size": float(np.mean(chunk_sizes)) if chunk_sizes else 0.0,
                "collection_name": self.collection_name,
                "distance_function": self.distance_function,
                "persist_directory": str(self.persist_directory),
            }
        else:
            stats = {
                "total_chunks": 0,
                "total_pdfs": 0,
                "unique_sources": [],
                "chunks_by_strategy": {},
                "chunks_by_level": {},
                "avg_coherence_score": 0.0,
                "avg_chunk_size": 0.0,
                "collection_name": self.collection_name,
                "distance_function": self.distance_function,
                "persist_directory": str(self.persist_directory),
            }
        return stats

    def reset_collection(self):
        """Reseta coleção (CUIDADO: deleta todos os dados)"""
        logger.warning(f"Resetando coleção '{self.collection_name}'")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Erro ao deletar coleção: {e}")
        self._initialize_collection()
        logger.info("Coleção resetada")

    def get_by_parent(self, parent_id: str) -> List[Dict]:
        """Busca todos os child chunks de um parent específico"""
        results = self.collection.get(
            where={"parent_id": parent_id}, include=["documents", "metadatas"]
        )
        formatted: List[Dict] = []
        for i in range(len(results["ids"])):
            formatted.append(
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )
        return formatted
