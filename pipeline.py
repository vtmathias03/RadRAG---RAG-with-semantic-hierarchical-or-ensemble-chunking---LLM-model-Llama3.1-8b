"""
Pipeline completo para processar documentos com RAG
Suporta as três estratégias de chunking
"""

import logging
from pathlib import Path
from typing import List, Dict
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

from core.advanced_chunker import AdvancedChunker
from core.chromadb_manager import ChromaDBManager
from core.pdf_processor import PDFProcessor
from core.llm_generator import OllamaGenerator
from core.reranker import CrossEncoderReranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CNENPipeline:
    """Pipeline completo para processar documentos"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa pipeline com configuração

        Args:
            config_path: Caminho para arquivo de configuração YAML
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Pipeline iniciado: {self.config['app']['name']} v{self.config['app']['version']}")

        # Inicializar componentes
        self._init_components()

    def _init_components(self):
        """Inicializa todos os componentes do pipeline"""
        logger.info("Inicializando componentes do pipeline...")

        # Modelo de embeddings
        logger.info(f"Carregando modelo: {self.config['embeddings']['model']}")
        device = self.config['embeddings'].get('device', 'cpu')
        if isinstance(device, str) and device.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning("CUDA device '%s' indisponível. Usando CPU.", device)
                device = 'cpu'
            else:
                device_index = 0
                if ':' in device:
                    try:
                        device_index = int(device.split(':', 1)[1])
                    except ValueError:
                        logger.warning("Não foi possível interpretar índice da GPU '%s'. Usando 0.", device)
                        device_index = 0
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.8, device=device_index)
                        logger.info("Limitando GPU cuda:%d a 80%% de memória disponível.", device_index)
                    except Exception as e:
                        logger.warning("Não foi possível limitar memória da GPU: %s", e)
                else:
                    logger.warning("Limitação de memória da GPU não suportada nesta versão do PyTorch.")
        self.embedding_model = SentenceTransformer(
            self.config['embeddings']['model'],
            device=device
        )
        normalize = self.config['embeddings'].get('normalize', False)
        batch_size = self.config['embeddings'].get('batch_size', 32)

        class PipelineEmbeddingFunction:
            def __init__(self, model, batch_size, normalize):
                self.model = model
                self.batch_size = batch_size
                self.normalize = normalize

            def _encode(self, items):
                if isinstance(items, str):
                    texts = [items]
                else:
                    texts = list(items)
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize
                )
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                if isinstance(embeddings, list):
                    if embeddings and isinstance(embeddings[0], (list, tuple)):
                        return [list(row) for row in embeddings]
                    return [list(embeddings)]
                return embeddings

            def __call__(self, input):
                return self._encode(input)

            def embed_query(self, input):
                return self._encode(input)


            def embed_documents(self, input):
                return self._encode(input)

            def name(self):
                return 'pipeline_sentence_transformer'

        chroma_embedding_function = PipelineEmbeddingFunction(
            self.embedding_model,
            batch_size=batch_size,
            normalize=normalize
        )

        # Chunker avançado
        chunking_config = self.config['advanced_chunking'].copy()
        strategy = chunking_config.pop('strategy')

        self.chunker = AdvancedChunker(
            embedding_model=self.embedding_model,
            chunking_strategy=strategy,
            **chunking_config
        )

        # ChromaDB - remover 'search' dos kwargs
        chromadb_config = self.config['chromadb'].copy()
        chromadb_config.pop('search', None)  # Remover 'search' se existir
        self.chromadb = ChromaDBManager(
            embedding_function=chroma_embedding_function,
            **chromadb_config
        )

        # PDF Processor
        self.pdf_processor = PDFProcessor(method='pymupdf')

        # LLM Generator
        self.llm = OllamaGenerator(**self.config['llm'])

        # Reranker (opcional)
        self.reranker = None
        if self.config.get('reranker', {}).get('enabled', False):
            try:
                logger.info("Inicializando reranker...")
                reranker_config = self.config['reranker']
                self.reranker = CrossEncoderReranker(
                    model=reranker_config['model'],
                    device=reranker_config.get('device', 'cpu'),
                    batch_size=reranker_config.get('batch_size', 32)
                )
                if self.reranker.ok:
                    logger.info("✓ Reranker inicializado com sucesso!")
                else:
                    logger.warning("⚠ Reranker não pôde ser carregado, continuando sem reranking")
                    self.reranker = None
            except Exception as e:
                logger.warning(f"Erro ao inicializar reranker: {e}. Continuando sem reranking.")
                self.reranker = None

        logger.info("Pipeline inicializado com sucesso!")

    def process_document(self, pdf_path: str) -> Dict:
        """
        Processa um documento PDF completo

        Args:
            pdf_path: Caminho para o arquivo PDF

        Returns:
            Dict com estatísticas do processamento
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Processando documento: {pdf_path.name}")

        if not pdf_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

        # 1. Extrair texto do PDF
        logger.info("Extraindo texto do PDF...")
        doc_data = self.pdf_processor.extract_structured_text(str(pdf_path))

        # 2. Criar chunks usando estratégia configurada
        logger.info("Criando chunks...")
        all_chunks = []

        for section in tqdm(doc_data['sections'], desc="Processando seções"):
            chunks = self.chunker.chunk_document(
                text=section['text'],
                metadata={
                    'source': pdf_path.name,
                    'section_title': section.get('title', ''),
                    'section_level': section.get('level', 0),
                    'pages': section.get('pages', ''),
                    **doc_data['metadata']
                }
            )
            all_chunks.extend(chunks)

        # 3. Adicionar ao ChromaDB
        logger.info("Adicionando chunks ao ChromaDB...")
        num_added = self.chromadb.add_chunks(all_chunks)

        # 4. Retornar estatísticas
        stats = {
            'document': pdf_path.name,
            'total_sections': len(doc_data['sections']),
            'total_pages': doc_data['total_pages'],
            'total_chunks': len(all_chunks),
            'parent_chunks': len([c for c in all_chunks if c.level == 'parent']),
            'child_chunks': len([c for c in all_chunks if c.level == 'child']),
            'chunks_inserted': num_added,
            'strategy_used': self.config['advanced_chunking']['strategy']
        }

        logger.info(f"Documento processado com sucesso!")
        logger.info(f"Estatísticas: {stats}")

        return stats

    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Processa todos os PDFs em um diretório

        Args:
            directory_path: Caminho para o diretório

        Returns:
            Lista de estatísticas de cada documento
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {directory}")

        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"Nenhum arquivo PDF encontrado em: {directory}")
            return []

        logger.info(f"Encontrados {len(pdf_files)} arquivos PDF")

        results = []
        for pdf_file in pdf_files:
            try:
                stats = self.process_document(str(pdf_file))
                results.append(stats)
            except Exception as e:
                logger.error(f"Erro ao processar {pdf_file.name}: {e}")
                results.append({
                    'document': pdf_file.name,
                    'error': str(e)
                })

        return results

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Busca documentos relevantes com reranking opcional

        Args:
            query: Texto da busca
            top_k: Número de resultados (padrão: config)

        Returns:
            Lista de resultados
        """
        if top_k is None:
            top_k = self.config['retrieval']['top_k_final']

        logger.info(f"Buscando: '{query}'")

        # Determinar quantos resultados buscar inicialmente (se reranker ativo, buscar mais)
        initial_k = top_k
        if self.reranker and self.reranker.ok:
            initial_k = self.config['reranker'].get('top_k_rerank', top_k * 10)
            logger.info(f"Reranker ativo: buscando {initial_k} resultados iniciais para reranking")

        # Gerar embedding da query
        query_embedding = self.embedding_model.encode(query)

        # Verificar se está usando ENSEMBLE
        chunking_strategy = self.config['advanced_chunking']['strategy']
        if chunking_strategy == 'semantic_hierarchical':
            # Busca para semantic_hierarchical: sem ponderação
            logger.info("Usando busca Semantic + Hierarchical (sem ponderação)")
            search_type = self.config['chromadb']['search']['type']

            if search_type == 'hybrid':
                results = self.chromadb.hybrid_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    n_results=initial_k,
                    alpha=self.config['chromadb']['search']['semantic_weight']
                )
            else:
                results = self.chromadb.search(
                    query_embedding=query_embedding,
                    n_results=initial_k
                )['results']
        else:
            # Busca normal (estratégia única: hierarchical, semantic, hybrid)
            search_type = self.config['chromadb']['search']['type']

            if search_type == 'hybrid':
                results = self.chromadb.hybrid_search(
                    query_text=query,
                    query_embedding=query_embedding,
                    n_results=initial_k,
                    alpha=self.config['chromadb']['search']['semantic_weight']
                )
            else:
                results = self.chromadb.search(
                    query_embedding=query_embedding,
                    n_results=initial_k
                )['results']

        # Aplicar reranking se disponível
        if self.reranker and self.reranker.ok and results:
            logger.info(f"Aplicando reranking em {len(results)} resultados...")
            results = self.reranker.rerank(query, results, text_key='text')
            logger.info(f"✓ Reranking concluído. Top score: {results[0].get('rerank_score', 0):.4f}")
            # Limitar ao top_k após reranking
            results = results[:top_k]

        # Filtrar por score mínimo
        min_score = self.config['retrieval'].get('min_relevance_score', 0.0)
        if min_score and results:
            filtered = []
            for res in results:
                score = res.get('final_score', res.get('score'))
                if score is None and 'distance' in res:
                    df = self.config['chromadb'].get('distance_function', 'cosine')
                    if df == 'cosine':
                        score = 1 - res['distance']
                if score is None or score >= min_score:
                    filtered.append(res)
            if not filtered:
                logger.info("Nenhum resultado acima do score mínimo, retornando lista vazia.")
            results = filtered

        logger.info(f"Encontrados {len(results)} resultados finais")

        return results

    def search_with_context(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Busca com contexto hierárquico completo

        Args:
            query: Texto da busca
            top_k: Número de resultados

        Returns:
            Lista de resultados com contexto
        """
        results = self.search(query, top_k)

        # Enriquecer com contexto do parent se configurado
        if self.config['retrieval']['use_parent_context']:
            for result in results:
                parent_id = result['metadata'].get('parent_id', '')

                if parent_id and parent_id != '':
                    # Buscar parent
                    try:
                        parent_results = self.chromadb.collection.get(
                            ids=[parent_id],
                            include=['documents']
                        )

                        if parent_results['documents']:
                            parent_text = parent_results['documents'][0]
                            result['full_context'] = (
                                f"CONTEXTO GERAL:\n{parent_text}\n\n"
                                f"DETALHE ESPECÍFICO:\n{result['text']}"
                            )
                        else:
                            result['full_context'] = result['text']
                    except Exception as e:
                        logger.warning(f"Erro ao buscar parent: {e}")
                        result['full_context'] = result['text']
                else:
                    result['full_context'] = result['text']

        return results

    def ask(self, query: str, top_k: int = None) -> Dict:
        """
        Faz uma pergunta e gera resposta usando LLM

        Args:
            query: Pergunta do usuário
            top_k: Número de chunks para contexto

        Returns:
            Dict com resposta e metadados
        """
        # 1. Buscar documentos relevantes
        logger.info(f"Buscando contexto para: {query}")
        context_chunks = self.search_with_context(query, top_k)

        if not context_chunks:
            return {
                'answer': "Não encontrei informações relevantes nos documentos para responder essa pergunta.",
                'sources': [],
                'num_chunks_used': 0
            }

        # 2. Gerar resposta com LLM
        logger.info(f"Gerando resposta com {len(context_chunks)} chunks de contexto")
        response = self.llm.generate_answer(query, context_chunks)

        return response

    def ask_stream(self, query: str, top_k: int = None):
        """
        Faz pergunta e retorna resposta em streaming

        Args:
            query: Pergunta do usuário
            top_k: Número de chunks para contexto

        Yields:
            Tokens da resposta
        """
        # Buscar contexto
        context_chunks = self.search_with_context(query, top_k)

        if not context_chunks:
            yield "Não encontrei informações relevantes nos documentos para responder essa pergunta."
            return

        # Gerar resposta em streaming
        for token in self.llm.generate_answer_stream(query, context_chunks):
            yield token

    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do sistema

        Returns:
            Dict com estatísticas
        """
        stats = self.chromadb.get_collection_stats()

        # Adicionar informações sobre o reranker
        if self.reranker and self.reranker.ok:
            stats['reranker_enabled'] = True
            stats['reranker_model'] = self.reranker.model_name
            stats['reranker_device'] = self.reranker.device
        else:
            stats['reranker_enabled'] = False

        return stats

    def reset(self):
        """Reseta o banco de dados (CUIDADO!)"""
        logger.warning("Resetando banco de dados...")
        self.chromadb.reset_collection()
        logger.info("Banco de dados resetado")


def main():
    """Exemplo de uso do pipeline"""

    # Criar pipeline
    pipeline = CNENPipeline("config.yaml")

    # Processar documentos
    data_dir = Path("data/pdfs")

    try:
        pdf_files = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
    except (PermissionError, OSError) as e:
        print(f"\n⚠️  Erro ao acessar diretório {data_dir}: {e}")
        pdf_files = []

    if pdf_files:
        print("\n" + "=" * 80)
        print("PROCESSANDO DOCUMENTOS")
        print("=" * 80)

        results = pipeline.process_directory(str(data_dir))

        print("\nResumo do processamento:")
        for result in results:
            if 'error' not in result:
                print(f"\n{result['document']}:")
                print(f"  Páginas: {result['total_pages']}")
                print(f"  Chunks: {result['total_chunks']}")
                print(f"  Parents: {result['parent_chunks']}")
                print(f"  Children: {result['child_chunks']}")
            else:
                print(f"\n{result['document']}: ERRO - {result['error']}")
    else:
        print("\nNenhum PDF encontrado em data/pdfs/")
        print("Adicione arquivos PDF ao diretório e execute novamente.")

    # Testar busca
    print("\n" + "=" * 80)
    print("TESTANDO BUSCA")
    print("=" * 80)

    queries = [
        "Quais são os limites de dose para trabalhadores?",
        "Como funciona o monitoramento com dosímetros?",
        "Quais são os princípios da proteção radiológica?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")

        results = pipeline.search_with_context(query, top_k=5)

        for i, result in enumerate(results, 1):
            score = result.get('final_score', result.get('score', 0))
            print(f"\n  [{i}] Score: {score:.3f}")
            print(f"      Fonte: {result['metadata'].get('source', 'N/A')}")
            print(f"      Seção: {result['metadata'].get('section_title', 'N/A')}")
            print(f"      Texto: {result['text'][:150]}...")

    # Estatísticas
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS DO SISTEMA")
    print("=" * 80)

    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()




