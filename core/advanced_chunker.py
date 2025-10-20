"""
Advanced Chunker com ENSEMBLE:
1. Hierarchical: Parent-child baseado em tamanho
2. Semantic: Clustering baseado em similaridade semântica
3. Hybrid: Combina hierarchical com semantic clustering
4. Semantic_hierarchical (ensemble): realiza hierarchical e semantic separadamente
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalChunk:
    """Representa um chunk hierárquico com contexto semântico"""
    chunk_id: str
    text: str
    level: str  # 'parent' ou 'child'
    strategy: str  # 'hierarchical', 'semantic', 'hybrid'
    parent_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)
    cluster_id: Optional[int] = None
    coherence_score: Optional[float] = None


class AdvancedChunker:
    """
    Chunker avançado:
    - hierarchical: Parent-child tradicional
    - semantic: Clustering semântico
    - hybrid: Combinação de hierarchical + semantic
    """

    def __init__(
        self,
        embedding_model,
        chunking_strategy: str = "hybrid",
        parent_chunk_size: int = 1000,
        parent_overlap: int = 200,
        child_chunk_size: int = 400,
        child_overlap: int = 100,
        similarity_threshold: float = 0.7,
        buffer_size: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
        sentence_splitter: str = "spacy",
        spacy_model: str = "pt_core_news_sm",
    ):
        """
        Args:
            embedding_model: Modelo para gerar embeddings
            chunking_strategy: 'hierarchical', 'semantic', 'hybrid' ou 'ensemble'
            parent_chunk_size: Tamanho dos parent chunks (tokens)
            parent_overlap: Overlap entre parents
            child_chunk_size: Tamanho dos child chunks
            child_overlap: Overlap entre children
            similarity_threshold: Threshold para clustering semântico
            buffer_size: Sentenças de buffer para contexto
            min_chunk_size: Tamanho mínimo do chunk
            max_chunk_size: Tamanho máximo do chunk
            sentence_splitter: Método de divisão de sentenças ('regex', 'nltk', 'spacy')
            spacy_model: Modelo spaCy usado quando sentence_splitter='spacy'
        """
        self.embedding_model = embedding_model
        self.chunking_strategy = chunking_strategy
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_splitter = (sentence_splitter or "regex").lower()
        self.spacy_model = spacy_model
        self._spacy_nlp = None
        self._nltk_tokenizer = None
        logger.info(f"AdvancedChunker iniciado com estratégia: {chunking_strategy}")


    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[HierarchicalChunk]:
        """
        Processa documento usando a estratégia configurada

        Args:
            text: Texto do documento
            metadata: Metadados do documento

        Returns:
            Lista de HierarchicalChunk objects
        """
        if metadata is None:
            metadata = {}

        if self.chunking_strategy == "hierarchical":
            return self._hierarchical_chunking(text, metadata)
        elif self.chunking_strategy == "semantic":
            return self._semantic_chunking(text, metadata)
        elif self.chunking_strategy == "hybrid":
            return self._hybrid_chunking(text, metadata)
        elif self.chunking_strategy == "semantic_hierarchical":
            return self._semantic_hierarchical_chunking(text, metadata)
        else:
            raise ValueError(f"Estratégia inválida: {self.chunking_strategy}")

    def _hierarchical_chunking(
        self,
        text: str,
        metadata: Dict
    ) -> List[HierarchicalChunk]:
        """
        Chunking hierárquico tradicional: parent-child baseado em tamanho
        """
        chunks = []
        words = text.split()

        # Criar parent chunks
        parent_chunks = []
        parent_step = max(1, self.parent_chunk_size - self.parent_overlap)
        for i in range(0, len(words), parent_step):
            parent_words = words[i:i + self.parent_chunk_size]
            if len(parent_words) < self.min_chunk_size:
                continue

            parent_id = f"hier_parent_{uuid.uuid4().hex[:8]}"
            parent_text = " ".join(parent_words)

            parent_embedding = self.embedding_model.encode(
                parent_text,
                convert_to_numpy=True
            )

            parent_chunk = HierarchicalChunk(
                chunk_id=parent_id,
                text=parent_text,
                level="parent",
                strategy="hierarchical",
                embedding=parent_embedding,
                metadata=metadata.copy()
            )
            parent_chunks.append(parent_chunk)
            chunks.append(parent_chunk)

        # Criar child chunks para cada parent
        for parent in parent_chunks:
            parent_words = parent.text.split()

            child_step = max(1, self.child_chunk_size - self.child_overlap)
            for i in range(0, len(parent_words), child_step):
                child_words = parent_words[i:i + self.child_chunk_size]

                if len(child_words) < self.min_chunk_size // 2:
                    continue

                child_id = f"hier_child_{uuid.uuid4().hex[:8]}"
                child_text = " ".join(child_words)

                child_embedding = self.embedding_model.encode(
                    child_text,
                    convert_to_numpy=True
                )

                child_chunk = HierarchicalChunk(
                    chunk_id=child_id,
                    text=child_text,
                    level="child",
                    strategy="hierarchical",
                    parent_id=parent.chunk_id,
                    embedding=child_embedding,
                    metadata=metadata.copy()
                )
                chunks.append(child_chunk)

        return chunks

    def _semantic_chunking(
        self,
        text: str,
        metadata: Dict
    ) -> List[HierarchicalChunk]:
        """
        Chunking semântico: clustering baseado em similaridade
        """
        # Dividir em sentenças
        sentences = self._split_into_sentences(text)

        if len(sentences) < 3:
            # Documento muito pequeno, retornar como chunk único
            chunk_id = f"sem_{uuid.uuid4().hex[:8]}"
            chunk_embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=True
            )
            return [HierarchicalChunk(
                chunk_id=chunk_id,
                text=text,
                level="parent",
                strategy="semantic",
                embedding=chunk_embedding,
                metadata=metadata.copy(),
                cluster_id=0,
                coherence_score=1.0
            )]

        # Gerar embeddings
        sentence_embeddings = self.embedding_model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False
        )

        # Clustering
        similarity_matrix = cosine_similarity(sentence_embeddings)
        clusters = self._perform_clustering(similarity_matrix, len(sentences))

        # Criar chunks a partir dos clusters
        chunks = self._create_chunks_from_clusters(
            sentences,
            sentence_embeddings,
            clusters,
            metadata,
            strategy="semantic"
        )

        return chunks

    def _hybrid_chunking(
        self,
        text: str,
        metadata: Dict
    ) -> List[HierarchicalChunk]:
        """
        Chunking híbrido: hierarchical + semantic clustering
        """
        chunks = []
        words = text.split()

        # 1. Criar parent chunks (hierárquico)
        parent_chunks = []
        parent_step = max(1, self.parent_chunk_size - self.parent_overlap)
        for i in range(0, len(words), parent_step):
            parent_words = words[i:i + self.parent_chunk_size]
            if len(parent_words) < self.min_chunk_size:
                continue

            parent_id = f"hyb_parent_{uuid.uuid4().hex[:8]}"
            parent_text = " ".join(parent_words)

            parent_embedding = self.embedding_model.encode(
                parent_text,
                convert_to_numpy=True
            )
            parent_chunk = HierarchicalChunk(
                chunk_id=parent_id,
                text=parent_text,
                level="parent",
                strategy="hybrid",
                embedding=parent_embedding,
                metadata=metadata.copy()
            )
            parent_chunks.append(parent_chunk)
            chunks.append(parent_chunk)

        # 2. Para cada parent, criar child chunks usando semantic clustering
        for parent in parent_chunks:
            sentences = self._split_into_sentences(parent.text)

            if len(sentences) < 2:
                # Muito poucas sentenças, criar child único
                child_id = f"hyb_child_{uuid.uuid4().hex[:8]}"
                child_embedding = self.embedding_model.encode(
                    parent.text,
                    convert_to_numpy=True
                )
                child_chunk = HierarchicalChunk(
                    chunk_id=child_id,
                    text=parent.text,
                    level="child",
                    strategy="hybrid",
                    parent_id=parent.chunk_id,
                    embedding=child_embedding,
                    metadata=metadata.copy(),
                    cluster_id=0,
                    coherence_score=1.0
                )
                chunks.append(child_chunk)
                continue

            # Gerar embeddings das sentenças
            sentence_embeddings = self.embedding_model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False
            )

            # Clustering semântico
            similarity_matrix = cosine_similarity(sentence_embeddings)
            clusters = self._perform_clustering(similarity_matrix, len(sentences))

            # Criar child chunks a partir dos clusters
            unique_clusters = np.unique(clusters)

            for cluster_id in unique_clusters:
                cluster_indices = np.where(clusters == cluster_id)[0]

                # Adicionar buffer
                expanded_indices = self._expand_with_buffer(
                    cluster_indices,
                    len(sentences)
                )

                # Sentenças do cluster
                cluster_sentences = [sentences[i] for i in expanded_indices]
                chunk_text = " ".join(cluster_sentences)

                # Calcular embedding e coerência
                chunk_embedding = np.mean(
                    sentence_embeddings[expanded_indices],
                    axis=0
                )
                coherence_score = self._calculate_coherence(
                    sentence_embeddings[expanded_indices]
                )

                # Verificar tamanho
                if self._is_valid_size(chunk_text):
                    child_id = f"hyb_child_{uuid.uuid4().hex[:8]}"
                    child_chunk = HierarchicalChunk(
                        chunk_id=child_id,
                        text=chunk_text,
                        level="child",
                        strategy="hybrid",
                        parent_id=parent.chunk_id,
                        embedding=chunk_embedding,
                        metadata=metadata.copy(),
                        cluster_id=int(cluster_id),
                        coherence_score=coherence_score
                    )
                    chunks.append(child_chunk)

        return chunks

    def _semantic_hierarchical_chunking(
        self,
        text: str,
        metadata: Dict
    ) -> List[HierarchicalChunk]:
        """
        Chunking Semantic + Hierarchical SEM ponderação:
        Gera chunks usando ambas estratégias, mas todos com mesmo peso na busca
        """
        logger.info("🎯 Executando Semantic + Hierarchical chunking (sem ponderação)...")

        all_chunks = []

        # 1. Processar com Hierarchical
        logger.info("  → Processando com Hierarchical...")
        hierarchical_chunks = self._hierarchical_chunking(text, metadata)
        for chunk in hierarchical_chunks:
            chunk.strategy = 'hierarchical'
            chunk.metadata['strategy'] = 'hierarchical'
            # SEM peso ensemble - todos chunks têm peso igual
        all_chunks.extend(hierarchical_chunks)
        logger.info(f"    ✓ {len(hierarchical_chunks)} chunks (hierarchical)")

        # 2. Processar com Semantic
        logger.info("  → Processando com Semantic...")
        semantic_chunks = self._semantic_chunking(text, metadata)
        for chunk in semantic_chunks:
            chunk.strategy = 'semantic'
            chunk.metadata['strategy'] = 'semantic'
            # SEM peso ensemble - todos chunks têm peso igual
        all_chunks.extend(semantic_chunks)
        logger.info(f"    ✓ {len(semantic_chunks)} chunks (semantic)")

        logger.info(f"✅ Semantic + Hierarchical: {len(all_chunks)} chunks totais criados")
        logger.info(f"   Hierarchical: {len(hierarchical_chunks)}")
        logger.info(f"   Semantic: {len(semantic_chunks)}")

        return all_chunks


    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças usando splitter configurado"""
        preferred = (self.sentence_splitter or 'regex').lower()
        ordered = []
        for candidate in (preferred, 'spacy', 'nltk', 'regex'):
            if candidate and candidate not in ordered:
                ordered.append(candidate)

        for splitter in ordered:
            if splitter == 'spacy':
                sentences = self._split_with_spacy(text)
            elif splitter == 'nltk':
                sentences = self._split_with_nltk(text)
            else:
                sentences = self._split_with_regex(text)

            if sentences:
                if splitter != preferred:
                    logger.debug("Sentence splitter fallback em uso: %s", splitter)
                return sentences

        stripped = text.strip()
        return [stripped] if stripped else []

    def _split_with_spacy(self, text: str) -> List[str]:
        """Tenta dividir sentenças usando spaCy"""
        try:
            if self._spacy_nlp is None:
                import spacy
                self._spacy_nlp = spacy.load(self.spacy_model)
            doc = self._spacy_nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as exc:
            logger.warning("spaCy indisponível para sentence splitting: %s", exc)
            self._spacy_nlp = None
            return []

    def _split_with_nltk(self, text: str) -> List[str]:
        """Tenta dividir sentenças usando NLTK"""
        try:
            if self._nltk_tokenizer is None:
                import nltk
                try:
                    self._nltk_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
                except LookupError as exc:
                    logger.warning("Tokenizador NLTK português indisponível: %s", exc)
                    self._nltk_tokenizer = None
                    return []
            sentences = [s.strip() for s in self._nltk_tokenizer.tokenize(text) if s.strip()]
            return sentences
        except Exception as exc:
            logger.warning("NLTK indisponível para sentence splitting: %s", exc)
            self._nltk_tokenizer = None
            return []

    def _split_with_regex(self, text: str) -> List[str]:
        """Divide texto em sentenças usando regex simples"""
        sentence_endings = re.compile(
            r"[.!?]\s+(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ])|"
            r"\n\n|"
            r"$"
        )
        sentences = sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        processed = []
        buffer = ''
        for sent in sentences:
            if len(sent.split()) < 5 and buffer:
                buffer += ' ' + sent
            else:
                if buffer:
                    processed.append(buffer)
                buffer = sent
        if buffer:
            processed.append(buffer)
        return processed

    def _perform_clustering(
        self,
        similarity_matrix: np.ndarray,
        n_sentences: int
    ) -> np.ndarray:
        """Aplica clustering hierárquico"""
        distance_matrix = 1 - similarity_matrix

        # Estimar número de clusters
        if n_sentences < 5:
            n_clusters = min(2, n_sentences)
        elif n_sentences < 15:
            n_clusters = n_sentences // 3
        else:
            n_clusters = n_sentences // 5

        n_clusters = max(2, min(n_clusters, n_sentences // 2))

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        clusters = clustering.fit_predict(distance_matrix)
        return clusters

    def _expand_with_buffer(
        self,
        indices: np.ndarray,
        max_index: int
    ) -> np.ndarray:
        """Expande índices com buffer"""
        expanded = set(indices)

        for idx in indices:
            for i in range(1, self.buffer_size + 1):
                if idx - i >= 0:
                    expanded.add(idx - i)
                if idx + i < max_index:
                    expanded.add(idx + i)

        return np.array(sorted(expanded))

    def _calculate_coherence(self, embeddings: np.ndarray) -> float:
        """Calcula score de coerência"""
        if len(embeddings) < 2:
            return 1.0

        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, 0)

        coherence = similarities.sum() / (len(embeddings) * (len(embeddings) - 1))
        return float(coherence)

    def _is_valid_size(self, text: str) -> bool:
        """Verifica tamanho válido"""
        tokens = len(text.split())
        return self.min_chunk_size <= tokens <= self.max_chunk_size

    def _create_chunks_from_clusters(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        clusters: np.ndarray,
        metadata: Dict,
        strategy: str = "semantic"
    ) -> List[HierarchicalChunk]:
        """Cria chunks a partir dos clusters"""
        chunks = []
        unique_clusters = np.unique(clusters)

        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            expanded_indices = self._expand_with_buffer(
                cluster_indices,
                len(sentences)
            )

            cluster_sentences = [sentences[i] for i in expanded_indices]
            chunk_text = " ".join(cluster_sentences)

            chunk_embedding = np.mean(embeddings[expanded_indices], axis=0)
            coherence_score = self._calculate_coherence(embeddings[expanded_indices])

            if self._is_valid_size(chunk_text):
                chunk_id = f"sem_{uuid.uuid4().hex[:8]}"
                chunk = HierarchicalChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    level="parent",
                    strategy=strategy,
                    embedding=chunk_embedding,
                    metadata=metadata.copy(),
                    cluster_id=int(cluster_id),
                    coherence_score=coherence_score
                )
                chunks.append(chunk)

        return chunks
