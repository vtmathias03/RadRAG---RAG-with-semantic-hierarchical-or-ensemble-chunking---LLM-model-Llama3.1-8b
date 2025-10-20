# -*- coding: utf-8 -*-
"""Interface de linha de comando para o sistema RAG."""

import os
import sys
from pathlib import Path
from pipeline import CNENPipeline


def _configure_console_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if os.name == "nt":
        try:
            os.system("chcp 65001 > nul")
        except Exception:
            pass


def print_header() -> None:
    print("\n" + "=" * 80)
    print("Sistema RAG - Semantic, Hierarchical e Hybrid Chunking")
    print("=" * 80 + "\n")


def print_menu() -> None:
    print("Menu:")
    print("  1. Processar PDF")
    print("  2. Processar diretório de PDFs")
    print("  3. Buscar documentos (sem LLM)")
    print("  4. Fazer pergunta (com LLM)")
    print("  5. Ver estatísticas")
    print("  6. Resetar banco de dados")
    print("  0. Sair\n")


def process_single_pdf(pipeline: CNENPipeline) -> None:
    pdf_path = input("\nCaminho do PDF: ").strip()

    if not Path(pdf_path).exists():
        print(f"⚠️  Arquivo não encontrado: {pdf_path}")
        return

    try:
        print("\nProcessando...")
        stats = pipeline.process_document(pdf_path)

        print("\n✅ Documento processado com sucesso!")
        print(f"   Páginas: {stats['total_pages']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Parents: {stats['parent_chunks']}")
        print(f"   Children: {stats['child_chunks']}")
        print(f"   Estratégia: {stats['strategy_used']}")

    except Exception as exc:
        print(f"❌ Erro ao processar: {exc}")


def process_directory(pipeline: CNENPipeline) -> None:
    dir_path = input("\nCaminho do diretório (padrão: data/pdfs): ").strip()

    if not dir_path:
        dir_path = "data/pdfs"

    if not Path(dir_path).exists():
        print(f"⚠️  Diretório não encontrado: {dir_path}")
        return

    try:
        print("\nProcessando PDFs...")
        results = pipeline.process_directory(dir_path)

        if not results:
            print("❌ Nenhum PDF encontrado no diretório")
            return

        print("\n✅ Processamento concluído!")
        print("\nResumo:")

        for result in results:
            if 'error' not in result:
                print(f"\n  {result['document']}:")
                print(f"    Páginas: {result['total_pages']}")
                print(f"    Chunks: {result['total_chunks']}")
            else:
                print(f"\n  {result['document']}: ❌ ERRO - {result['error']}")

    except Exception as exc:
        print(f"❌ Erro ao processar diretório: {exc}")


def search_documents(pipeline: CNENPipeline) -> None:
    print("\nBuscar documentos (sem LLM)")

    query = input("\nDigite sua pergunta: ").strip()

    if not query:
        print("⚠️  Pergunta vazia")
        return

    try:
        top_k = input("Número de resultados (padrão: 5): ").strip()
        top_k = int(top_k) if top_k else 5

        print("\nBuscando...")
        results = pipeline.search_with_context(query, top_k=top_k)

        if not results:
            print("\nNenhum resultado encontrado")
            return

        print(f"\n✅ {len(results)} resultados encontrados:\n")

        for idx, result in enumerate(results, 1):
            score = result.get('final_score', result.get('score', 0))
            print("-" * 80)
            print(f"[{idx}] Score: {score:.3f}")

            # Mostrar scores do reranker se disponível
            if 'rerank_score' in result:
                retriever_score = result.get('retriever_score', 0)
                print(f"    Retriever Score: {retriever_score:.3f}")
                print(f"    Rerank Score: {result['rerank_score']:.3f}")

            print(f"    Fonte: {result['metadata'].get('source', 'N/A')}")
            print(f"    Seção: {result['metadata'].get('section_title', 'N/A')}")
            print(f"    Nível: {result['metadata'].get('level', 'N/A')}")
            print("    Texto:")
            print(f"    {result['text'][:300]}...")

            if result.get('full_context') and result['full_context'] != result['text']:
                print("    Contexto adicional disponível")

        print("-" * 80 + "\n")

    except Exception as exc:
        print(f"❌ Erro na busca: {exc}")


def ask_question(pipeline: CNENPipeline) -> None:
    print("\nFazer pergunta (com LLM)")

    query = input("\nDigite sua pergunta: ").strip()

    if not query:
        print("⚠️  Pergunta vazia")
        return

    try:
        top_k = input("Número de documentos para contexto (padrão: 5): ").strip()
        top_k = int(top_k) if top_k else 5

        import time
        start_time = time.perf_counter()
        print("\nGerando resposta...")
        response = pipeline.ask(query, top_k=top_k)
        elapsed = time.perf_counter() - start_time

        print("\n" + "-" * 80)
        print(f"RESPOSTA (em {elapsed:.3f}s):")
        print("-" * 80 + "\n")
        print(response['answer'])

        print("\n" + "-" * 80)
        print(f"FONTES ({response['num_chunks_used']} documentos usados):")
        print("-" * 80 + "\n")

        for idx, source in enumerate(response['sources'], 1):
            print(f"[{idx}] {source['source']} (Score: {source.get('score', 0):.3f})")
            print(f"    {source['text']}\n")

        print("-" * 80 + "\n")

    except Exception as exc:
        print(f"❌ Erro ao gerar resposta: {exc}")


def show_stats(pipeline: CNENPipeline) -> None:
    try:
        print("\nEstatísticas do Sistema")
        print("-" * 80)

        stats = pipeline.get_stats()

        print(f"\n  PDFs processados: {stats['total_pdfs']}")
        print(f"  Chunks disponíveis para consulta: {stats['total_chunks']}")

        if stats['chunks_by_strategy']:
            print("\n  Chunks por estratégia:")
            for strategy, count in stats['chunks_by_strategy'].items():
                print(f"     {strategy}: {count} chunks")

        if stats['chunks_by_level']:
            print("\n  Chunks por nível:")
            for level, count in stats['chunks_by_level'].items():
                print(f"     {level}: {count} chunks")

        if stats['unique_sources']:
            print("\n  PDFs no sistema:")
            for idx, source in enumerate(stats['unique_sources'], 1):
                print(f"     {idx}. {source}")

        print("\n  Métricas de qualidade:")
        print(f"     Coerência média: {stats['avg_coherence_score']:.3f}")
        print(f"     Tamanho médio dos chunks: {stats['avg_chunk_size']:.1f} tokens")

        print("\n  Configuração:")
        print(f"     Coleção: {stats['collection_name']}")
        print(f"     Função de distância: {stats['distance_function']}")
        print(f"     Diretório: {stats['persist_directory']}")

        # Informações sobre o reranker
        if stats.get('reranker_enabled'):
            print("\n  Reranker:")
            print(f"     Status: ✓ Ativo")
            print(f"     Modelo: {stats.get('reranker_model', 'N/A')}")
            print(f"     Device: {stats.get('reranker_device', 'N/A')}")
        else:
            print("\n  Reranker:")
            print(f"     Status: ✗ Desativado")

        print("\n" + "-" * 80)

    except Exception as exc:
        print(f"❌ Erro ao obter estatísticas: {exc}")


def reset_database(pipeline: CNENPipeline) -> None:
    print("\n⚠️  ATENÇÃO: Esta ação vai apagar todos os dados do ChromaDB!")
    confirm = input("Digite 'CONFIRMAR' para continuar: ").strip()

    if confirm != "CONFIRMAR":
        print("Operação cancelada")
        return

    try:
        print("\nLimpando base de dados...")
        pipeline.reset()
        print("✅ Base de dados resetada com sucesso!")
    except Exception as exc:
        print(f"❌ Erro ao resetar: {exc}")


def main() -> None:
    _configure_console_utf8()
    print_header()

    try:
        print("Inicializando pipeline...")
        pipeline = CNENPipeline("config.yaml")
        print("✅ Pipeline inicializado!\n")
    except Exception as exc:
        print(f"❌ Erro ao inicializar pipeline: {exc}")
        return

    while True:
        print_menu()
        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            process_single_pdf(pipeline)
        elif choice == "2":
            process_directory(pipeline)
        elif choice == "3":
            search_documents(pipeline)
        elif choice == "4":
            ask_question(pipeline)
        elif choice == "5":
            show_stats(pipeline)
        elif choice == "6":
            reset_database(pipeline)
        elif choice == "0":
            print("\nAté logo!\n")
            break
        else:
            print("\nOpção inválida!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário. Até logo!\n")
    except Exception as exc:
        print(f"\n❌ Erro fatal: {exc}\n")
