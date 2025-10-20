"""
Reranker baseado em CrossEncoder (Sentence-Transformers)
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder 
except Exception as e:  
    CrossEncoder = None  
    logger.warning("CrossEncoder indisponível: %s", e)

try:
    import torch  
except Exception: 
    torch = None  


class CrossEncoderReranker:
    """Wrapper simples para reranking com CrossEncoder."""

    def __init__(self, model: str, device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        # Fallback para CPU se CUDA indisponível
        try:
            if isinstance(self.device, str) and self.device.startswith('cuda') and (not torch or not torch.cuda.is_available()):
                logger.warning("CUDA '%s' indisponível; usando CPU.", self.device)
                self.device = 'cpu'
        except Exception:
            self.device = 'cpu'
        self.model = None
        self.ok = False
        try:
            if CrossEncoder is None:
                raise RuntimeError("sentence-transformers CrossEncoder não disponível")
            self.model = CrossEncoder(model, device=self.device)
            self.ok = True
            logger.info("Reranker carregado: %s em %s", model, self.device)
        except Exception as e:
            logger.warning("Falha ao carregar CrossEncoder '%s': %s", model, e)
            self.ok = False

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        if not self.ok or not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(  
            pairs,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        try:
            return scores.tolist()  
        except Exception:
            return list(scores)  

    def rerank(self, query: str, results: List[Dict], text_key: str = 'text') -> List[Dict]:
        if not self.ok or not results:
            return results
        texts = [r.get(text_key, "") for r in results]
        scores = self.score_pairs(query, texts)
        out: List[Dict] = []
        for r, s in zip(results, scores):
            r2 = dict(r)
            r2['retriever_score'] = r.get('final_score', r.get('score', None))
            r2['rerank_score'] = float(s)
            r2['final_score'] = float(s)
            out.append(r2)
        out.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        return out
