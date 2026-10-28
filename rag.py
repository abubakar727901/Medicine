"""Retrieve relevant medicine knowledge using cosine similarity."""

from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import KnowledgeChunk, get_embeddings


@dataclass
class SearchResult:
    chunk: KnowledgeChunk
    score: float


def search(
    question: str,
    chunks: list[KnowledgeChunk],
    embeddings: np.ndarray,
    client: OpenAI,
    top_k: int = 3,
    threshold: float = 0.25,
) -> list[SearchResult]:
    """Return the highest-scoring chunks, excluding weak matches."""
    if not question.strip():
        raise ValueError("Enter a question before searching.")
    if not chunks or embeddings.size == 0:
        raise ValueError("The medicine knowledge base is empty.")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Knowledge chunks and embeddings do not match.")
    if top_k < 1:
        raise ValueError("The number of results must be positive.")

    question_embedding = get_embeddings([question], client)
    scores = cosine_similarity(question_embedding, embeddings)[0]
    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        SearchResult(chunk=chunks[index], score=float(scores[index]))
        for index in ranked_indices
        if scores[index] >= threshold
    ]


def is_personalized_medical_request(question: str) -> bool:
    """Detect requests for individualized treatment, dosing, or diagnosis."""
    text = question.lower()
    patterns = (
        r"\b(?:how many|how much|what dose|dosage|dose of|mg should i|"
        r"tablets? should i|pills? should i|how often should i)\b",
        r"\b(?:should i|can i|may i|is it safe for me to|can my child|"
        r"can i give|should my child)\b.{0,100}\b(?:take|use|start|stop|"
        r"change|combine|mix|medicine|medication|drug|tablet|pill|dose)\b",
        r"\b(?:take|use|start|stop|change|combine|mix)\b.{0,100}\b"
        r"(?:while pregnant|during pregnancy|while breastfeeding|with my "
        r"(?:condition|medication|medicine|symptoms)|for my symptoms)\b",
        r"\b(?:what medicine|which medicine|what medication|which medication)"
        r"\b.{0,100}\b(?:should i|for my|for this|for these symptoms)\b",
        r"\b(?:diagnose|what disease do i have|what is wrong with me)\b",
        r"\b(?:i have|i am having|my symptoms are)\b.{0,100}\b"
        r"(?:what should i take|which medicine|what medicine|treatment)\b",
    )
    import re

    return any(re.search(pattern, text) for pattern in patterns)
