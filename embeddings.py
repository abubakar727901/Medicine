"""Create searchable medicine text and OpenAI embeddings."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from openai import OpenAI


@dataclass
class KnowledgeChunk:
    medicine: str
    topic: str
    content: str
    sources: list[dict[str, str]]


def make_chunks(records: list[dict[str, Any]], size: int = 800) -> list[KnowledgeChunk]:
    """Convert medicine records into short, labeled chunks for retrieval."""
    if size < 1:
        raise ValueError("Chunk size must be positive.")

    chunks = []
    fields = [
        ("Overview", "generic_name"),
        ("Active ingredient", "active_ingredient"),
        ("Chemical formula", "chemical_formula"),
        ("Medicine class", "medicine_class"),
        ("General purpose", "general_purpose"),
        ("Common uses", "common_uses"),
        ("General action", "general_action"),
        ("General warnings", "general_warnings"),
        ("Common side effects", "common_side_effects"),
        ("Availability", "availability_note"),
    ]

    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every medicine record must have a name.")

        for topic, field in fields:
            value = record.get(field)
            if isinstance(value, list):
                text = "; ".join(str(item) for item in value if item)
            elif isinstance(value, str):
                text = value.strip()
            else:
                text = ""

            if not text:
                continue

            labeled_text = f"Medicine: {name}\n{topic}: {text}"
            for start in range(0, len(labeled_text), size):
                chunks.append(
                    KnowledgeChunk(
                        medicine=name,
                        topic=topic,
                        content=labeled_text[start : start + size],
                        sources=record.get("sources", []),
                    )
                )

    if not chunks:
        raise ValueError("The medicine knowledge base contains no searchable information.")
    return chunks


def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    """Get one embedding vector per text using OpenAI's embedding API."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Embedding input must contain non-empty text.")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return np.array([item.embedding for item in response.data], dtype=float)
