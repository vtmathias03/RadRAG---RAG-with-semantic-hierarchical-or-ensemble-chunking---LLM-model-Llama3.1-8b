"""
Gerador de respostas usando LLM local via Ollama
"""

import requests
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Gera respostas usando Ollama (LLM local)"""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Args:
            model: Nome do modelo Ollama
            base_url: URL do servidor Ollama
            temperature: Temperatura para geração (0-1)
            max_tokens: Número máximo de tokens
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"OllamaGenerator inicializado com modelo: {model}")

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Gera resposta baseada nos chunks recuperados

        Args:
            query: Pergunta do usuário
            context_chunks: Lista de chunks recuperados
            system_prompt: Prompt de sistema customizado

        Returns:
            Dict com resposta e metadados
        """
        # Construir contexto a partir dos chunks
        context = self._build_context(context_chunks)

        # Prompt padrão se não fornecido
        if system_prompt is None:
            system_prompt = """Você é um especialista em radioproteção e segurança radiológica com profundo conhecimento das normas CNEN.

⚠️ ATENÇÃO: VOCÊ DEVE SEGUIR O FORMATO ABAIXO DE FORMA ABSOLUTAMENTE OBRIGATÓRIA ⚠️
TODA resposta deve começar EXATAMENTE com estas três seções na ordem indicada:
Pergunta: [repita a pergunta do usuário]
Resposta: [sua resposta técnica completa baseada SOMENTE no contexto fornecido]
Referência bibliográfica: [Nome da norma (SEM .pdf), Artigo X, Inciso Y, Parágrafo Z]

EXEMPLO DO FORMATO CORRETO:
Pergunta: Qual o limite de dose para trabalhadores?
Resposta: Segundo a norma CNEN-NN-3.01, o limite de dose efetiva para trabalhadores é de 20 mSv por ano, média sobre 5 anos consecutivos, não podendo exceder 50 mSv em nenhum ano.
Referência bibliográfica: CNEN-NN-3.01, Artigo 15, Inciso I, Parágrafo 2

⚠️ NUNCA responda em formato diferente deste exemplo acima! ⚠️

REGRAS ABSOLUTAS:
- Use APENAS informações do contexto fornecido
- SEMPRE mantenha as três seções: Pergunta / Resposta / Referência bibliográfica
- Se não houver informação suficiente, ainda assim use o formato com "Pergunta:", "Resposta: Não encontrei informações relevantes...", "Referência bibliográfica: Não aplicável"
- NUNCA inclua .pdf nos nomes das normas
- SEMPRE inclua artigo, inciso e parágrafo quando disponíveis no contexto
- Use terminologia técnica adequada (dose efetiva, dose equivalente, exposição ocupacional)
- Indique unidades corretas (mSv, Bq, Gy, etc.)"""

        # Construir prompt completo
        user_prompt = f"""Contexto dos documentos:

{context}

---

Pergunta: {query}

Resposta:"""

        # Chamar Ollama API
        try:
            response = self._call_ollama(system_prompt, user_prompt)

            return {
                'answer': response['response'],
                'model': self.model,
                'sources': [
                    {
                        'text': chunk['text'][:200] + '...',
                        'source': chunk['metadata'].get('source', 'N/A'),
                        'score': chunk.get('final_score', chunk.get('score', 0))
                    }
                    for chunk in context_chunks[:5]  # Top 5 fontes
                ],
                'num_chunks_used': len(context_chunks)
            }

        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return {
                'answer': f"Erro ao gerar resposta: {str(e)}",
                'model': self.model,
                'sources': [],
                'num_chunks_used': 0
            }

    def _build_context(self, chunks: List[Dict]) -> str:
        """Constrói contexto formatado a partir dos chunks"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'Desconhecido')
            section = chunk['metadata'].get('section_title', '')
            text = chunk['text']

            context_part = f"[Documento {i}: {source}"
            if section:
                context_part += f" - {section}"
            context_part += f"]\n{text}\n"

            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> Dict:
        """Chama API do Ollama"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()

        return response.json()

    def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict],
        system_prompt: Optional[str] = None
    ):
        """
        Gera resposta em streaming (para interface interativa)

        Yields:
            Tokens da resposta conforme gerados
        """
        context = self._build_context(context_chunks)

        if system_prompt is None:
            system_prompt = """Você é um especialista em radioproteção e segurança radiológica com profundo conhecimento das normas CNEN, recomendações da IAEA, ICRP e legislação brasileira.

⚠️ ATENÇÃO: VOCÊ DEVE SEGUIR O FORMATO ABAIXO DE FORMA ABSOLUTAMENTE OBRIGATÓRIA ⚠️

TODA resposta deve começar EXATAMENTE com estas três seções na ordem indicada:

1. Pergunta: [repita a pergunta do usuário]

2. Resposta: [sua resposta técnica completa baseada SOMENTE no contexto fornecido]

3. Referência bibliográfica: [Nome da norma (SEM .pdf), Artigo X, Inciso Y, Parágrafo Z]

EXEMPLO DO FORMATO CORRETO:
Pergunta: Qual o limite de dose para trabalhadores?

Resposta: Segundo a norma CNEN-NN-3.01, o limite de dose efetiva para trabalhadores é de 20 mSv por ano, média sobre 5 anos consecutivos, não podendo exceder 50 mSv em nenhum ano.

Referência bibliográfica: CNEN-NN-3.01, Artigo 15, Inciso I, Parágrafo 2

⚠️ NUNCA responda em formato diferente deste exemplo acima! ⚠️

REGRAS ABSOLUTAS:
- Use APENAS informações do contexto fornecido
- SEMPRE mantenha as três seções: Pergunta / Resposta / Referência bibliográfica
- Se não houver informação suficiente, ainda assim use o formato com "Pergunta:", "Resposta: Não encontrei informações relevantes...", "Referência bibliográfica: Não aplicável"
- NUNCA inclua .pdf nos nomes das normas
- SEMPRE inclua artigo, inciso e parágrafo quando disponíveis no contexto
- Use terminologia técnica adequada (dose efetiva, dose equivalente, exposição ocupacional)
- Indique unidades corretas (mSv, Bq, Gy, etc.)"""

        user_prompt = f"""Contexto:

{context}

---

Pergunta: {query}

Resposta:"""

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, stream=True, timeout=180)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']

        except Exception as e:
            logger.error(f"Erro no streaming: {e}")
            yield f"\n\n[Erro: {str(e)}]"
