"""
Processador de PDFs para extração estruturada de texto
Backend único: PyMuPDF (fitz)
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Processa arquivos PDF e extrai texto estruturado
    """

    def __init__(self, method: str = "pymupdf"):
        """
        Args:
            method: 'pymupdf' (fitz)
        """
        self.method = 'pymupdf'
        if method != 'pymupdf':
            logger.warning(f"PDFProcessor: método '{method}' solicitado, usando 'pymupdf' (único suportado)")
        logger.info("PDFProcessor iniciado com método: pymupdf")

    def extract_text(self, pdf_path: str) -> str:
        """
        Extrai texto simples do PDF

        Args:
            pdf_path: Caminho para o arquivo PDF

        Returns:
            Texto extraído
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
        logger.info(f"Extraindo texto de: {pdf_path.name}")
        return self._extract_with_pymupdf(pdf_path)

    def extract_structured_text(self, pdf_path: str) -> Dict:
        """
        Extrai texto estruturado com metadados
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
        logger.info(f"Extraindo texto estruturado de: {pdf_path.name}")
        return self._extract_structured_pymupdf(pdf_path)
    # suporte a PyPDF2 removido — use _extract_with_pymupdf()

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extrai texto usando PyMuPDF"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF não instalado. Execute: pip install PyMuPDF")

        text = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text()
                    if page_text:
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Erro ao extrair página {page_num}: {e}")

        full_text = "\n\n".join(text)
        logger.info(f"Extraído: {len(full_text)} caracteres")

        return full_text

    def _extract_structured_pymupdf(self, pdf_path: Path) -> Dict:
        """Extrai texto estruturado usando PyMuPDF"""
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF não instalado. Execute: pip install PyMuPDF")

        sections = []
        metadata = {}

        with fitz.open(pdf_path) as doc:
            # Extrair metadados
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', '')
            }
            total_pages = len(doc)

            # Tentar extrair TOC (table of contents)
            toc = doc.get_toc()

            if toc:
                # Se há TOC, usar estrutura de capítulos
                sections = self._extract_by_toc(doc, toc)
            else:
                # Senão, extrair por página
                for page_num, page in enumerate(doc, 1):
                    try:
                        page_text = page.get_text()
                        if page_text and page_text.strip():
                            sections.append({
                                'title': f"Página {page_num}",
                                'text': page_text,
                                'level': 0,
                                'pages': str(page_num)
                            })
                    except Exception as e:
                        logger.warning(f"Erro ao extrair página {page_num}: {e}")

        return {
            'metadata': metadata,
            'sections': sections,
            'total_pages': total_pages
        }

    def _extract_by_toc(self, doc, toc: List) -> List[Dict]:
        """Extrai texto usando Table of Contents"""
        sections = []

        for i, (level, title, page_num) in enumerate(toc):
            # Determinar páginas da seção
            start_page = page_num - 1  # fitz usa índice 0-based

            # Próxima seção do mesmo nível ou menor
            end_exclusive = len(doc)
            for j in range(i + 1, len(toc)):
                if toc[j][0] <= level:
                    end_exclusive = toc[j][2] - 1
                    break

            # Blindagem para TOC malformada
            stop_idx = min(end_exclusive, len(doc))
            if stop_idx <= start_page:
                continue

            # Extrair texto das páginas da seção
            # Extrair texto das páginas da seção
            section_text = []
            for page_idx in range(start_page, stop_idx):
                try:
                    page = doc[page_idx]
                    page_text = page.get_text()
                    if page_text:
                        section_text.append(page_text)
                except Exception as e:
                    logger.warning(f"Erro ao extrair página {page_idx + 1}: {e}")

            if section_text:
                sections.append({
                    'title': title,
                    'text': "\n\n".join(section_text),
                    'level': level,
                    'pages': f"{start_page + 1}-{stop_idx}"
                })

        return sections

    def extract_pages_range(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int
    ) -> str:
        """
        Extrai texto de um range de páginas

        Args:
            pdf_path: Caminho para o PDF
            start_page: Página inicial (1-indexed)
            end_page: Página final (1-indexed, inclusive)

        Returns:
            Texto extraído
        """
        return self._extract_range_pymupdf(pdf_path, start_page, end_page)

    def _extract_range_pymupdf(
        self,
        pdf_path: str,
        start_page: int,
        end_page: int
    ) -> str:
        """Extrai range de páginas com PyMuPDF"""
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF não instalado")

        text = []

        with fitz.open(pdf_path) as doc:
            for page_num in range(start_page - 1, min(end_page, len(doc))):
                try:
                    page_text = doc[page_num].get_text()
                    if page_text:
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Erro ao extrair página {page_num + 1}: {e}")

        return "\n\n".join(text)



