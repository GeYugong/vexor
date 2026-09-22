"""Compare single and batched retrieval using a real local embedding model.

Only synthetic documents are indexed. Indexing and model warmup are excluded
from the measurements; each arm uses its own cold query cache. No user config
or existing index is changed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal, TypeVar
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from vexor import RecordResult, VexorClient
from vexor.providers.local import LocalEmbeddingBackend
from vexor.services.search_service import SearchResponse

T = TypeVar("T")
BenchmarkResponse = SearchResponse | list[RecordResult]
ARMS: tuple[Literal["single", "batch"], ...] = ("single", "batch")

DOCUMENTS = {
    "authentication.txt": "Validate login passwords and issue authentication tokens.",
    "database.txt": "Open database connections and execute SQL transactions safely.",
    "cache.txt": "Cache query results in memory and expire stale cached entries.",
}
QUERIES = [
    "validate login passwords", "execute SQL transactions", "expire cached entries",
    "issue authentication tokens", "database connections", "cache query results",
]


def measure(operation: Callable[[], T], *, label: str) -> T:
    """Measure elapsed time and real embedding calls without replacing their results."""
    calls: list[int] = []
    original = LocalEmbeddingBackend.embed

    def counted(backend: LocalEmbeddingBackend, texts: Sequence[str]) -> NDArray[np.float32]:
        """Record the request size and invoke the real local backend."""
        calls.append(len(texts))
        return original(backend, texts)

    with patch.object(LocalEmbeddingBackend, "embed", counted):
        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
    print(f"{label}: seconds={elapsed:.3f} embedding_calls={len(calls)} batch_sizes={calls}")
    return result


def assert_equivalent(
    single: Sequence[BenchmarkResponse], batch: Sequence[BenchmarkResponse]
) -> None:
    """Compare ordered identities, scores, and source data for each query."""
    assert len(single) == len(batch) == len(QUERIES)
    for left, right in zip(single, batch, strict=True):
        if isinstance(left, SearchResponse):
            assert isinstance(right, SearchResponse)
            keys_left = [item.path.name for item in left.results]
            keys_right = [item.path.name for item in right.results]
            scores_left = [item.score for item in left.results]
            scores_right = [item.score for item in right.results]
            assert [item.content for item in left.results] == [
                item.content for item in right.results
            ]
        else:
            assert isinstance(right, list)
            keys_left = [item.id for item in left]
            keys_right = [item.id for item in right]
            scores_left = [item.score for item in left]
            scores_right = [item.score for item in right]
            assert [item.metadata for item in left] == [item.metadata for item in right]
        assert keys_left == keys_right, (keys_left, keys_right)
        np.testing.assert_allclose(scores_left, scores_right, atol=1e-5)


def measure_queries(
    single: Callable[[str], T],
    batch: Callable[[Sequence[str]], list[T]],
    arm: Literal["single", "batch"],
    label: str,
) -> list[T]:
    """Measure either query strategy using callables with options already bound."""
    def operation() -> list[T]:
        """Run the selected strategy against the same fixed query list."""
        if arm == "single":
            return [single(query) for query in QUERIES]
        return batch(QUERIES)

    return measure(operation, label=label)


def run(model: str) -> None:
    """Compare five ranking/surface combinations using isolated synthetic corpora."""
    with TemporaryDirectory(prefix="vexor-batch-") as temporary:
        base = Path(temporary)
        root = base / "documents"
        root.mkdir()
        for name, text in DOCUMENTS.items():
            (root / name).write_text(text + "\n", encoding="utf-8")
        config = {"provider": "local", "model": model, "rerank": "off"}
        for rerank in ("off", "hybrid"):
            config["rerank"] = rerank
            file_arms: dict[str, list[SearchResponse]] = {}
            for arm in ARMS:
                with VexorClient(cache_dir=base / f"{rerank}-{arm}") as client:
                    client.set_config_json(config, replace=True)
                    client.index(path=root, mode="full")
                    file_arms[arm] = measure_queries(
                        partial(client.search, path=root, mode="full", include_content=True),
                        partial(client.search_many, path=root, mode="full", include_content=True),
                        arm,
                        f"files/{rerank}/{arm}",
                    )
            assert_equivalent(file_arms["single"], file_arms["batch"])

        with VexorClient(cache_dir=base / "memory") as client:
            client.set_config_json(config, replace=True)
            index = client.index_in_memory(path=root, mode="full")
            single = measure(lambda: [index.search(q, include_content=True) for q in QUERIES],
                             label="memory/hybrid/single")
            batch = measure(lambda: index.search_many(QUERIES, include_content=True),
                            label="memory/hybrid/batch")
            assert_equivalent(single, batch)

        for rerank in ("off", "hybrid"):
            record_arms: dict[str, list[list[RecordResult]]] = {}
            for arm in ARMS:
                with VexorClient(cache_dir=base / f"records-{rerank}-{arm}") as client:
                    client.set_config_json(config, replace=True)
                    handle = client.collection("documents")
                    handle.upsert_many([
                        {"id": name, "text": text, "metadata": {"tenant": "allowed"}}
                        for name, text in DOCUMENTS.items()
                    ] + [{"id": "private", "text": "private authentication token",
                          "metadata": {"tenant": "other"}}])
                    record_arms[arm] = measure_queries(
                        partial(handle.search, filters={"tenant": "allowed"}, rerank=rerank),
                        partial(handle.search_many, filters={"tenant": "allowed"}, rerank=rerank),
                        arm, f"collections/{rerank}/{arm}",
                    )
            assert_equivalent(record_arms["single"], record_arms["batch"])
    print("PASS: file, memory, and filtered collection batches match single-query results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    run(parser.parse_args().model)
