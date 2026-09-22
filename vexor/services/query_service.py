"""Validation shared by batched file and collection queries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..text import Messages


def normalize_queries(queries: Sequence[str]) -> list[str]:
    """Validate the entire input before indexing, embedding, or opening a store."""

    if isinstance(queries, (str, bytes)) or not isinstance(queries, Sequence):
        raise ValueError(Messages.ERROR_QUERIES_SEQUENCE)
    cleaned: list[str] = []
    for position, query in enumerate(queries):
        if not isinstance(query, str) or not query.strip():
            raise ValueError(Messages.ERROR_QUERY_ITEM.format(position=position))
        cleaned.append(query.strip())
    return cleaned


def validate_embedding_vectors(vectors: object, count: int) -> np.ndarray:
    """Reject incomplete or non-finite provider batches before caching any row."""

    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(Messages.ERROR_QUERY_VECTORS.format(count=count)) from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] != count
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(Messages.ERROR_QUERY_VECTORS.format(count=count))
    return matrix
