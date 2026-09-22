"""Compare single and batched retrieval using a real local embedding model.

Only synthetic documents are indexed. Indexing and model warmup are excluded
from the measurements; each arm uses its own cold query cache. No user config
or existing index is changed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np

from vexor import VexorClient
from vexor.providers.local import LocalEmbeddingBackend

DOCUMENTS = {
    "authentication.txt": "Validate login passwords and issue authentication tokens.",
    "database.txt": "Open database connections and execute SQL transactions safely.",
    "cache.txt": "Cache query results in memory and expire stale cached entries.",
}
QUERIES = [
    "validate login passwords", "execute SQL transactions", "expire cached entries",
    "issue authentication tokens", "database connections", "cache query results",
]


def measure(operation: Callable, *, label: str) -> object:
    calls: list[int] = []
    original = LocalEmbeddingBackend.embed

    def counted(backend: LocalEmbeddingBackend, texts: Sequence[str]) -> np.ndarray:
        calls.append(len(texts))
        return original(backend, texts)

    with patch.object(LocalEmbeddingBackend, "embed", counted):
        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
    print(f"{label}: seconds={elapsed:.3f} embedding_calls={len(calls)} batch_sizes={calls}")
    return result


def assert_equivalent(single: list, batch: list, *, records: bool = False) -> None:
    assert len(single) == len(batch) == len(QUERIES)
    for left, right in zip(single, batch, strict=True):
        if not records:
            left, right = left.results, right.results
        keys_left = [item.id if records else item.path.name for item in left]
        keys_right = [item.id if records else item.path.name for item in right]
        assert keys_left == keys_right, (keys_left, keys_right)
        np.testing.assert_allclose(
            [item.score for item in left], [item.score for item in right], atol=1e-5,
        )
        if records:
            assert [item.metadata for item in left] == [item.metadata for item in right]
        else:
            assert [item.content for item in left] == [item.content for item in right]


def measure_queries(target, arm: str, options: dict, label: str) -> list:
    def operation() -> list:
        if arm == "single":
            return [target.search(query, **options) for query in QUERIES]
        return target.search_many(QUERIES, **options)

    return measure(operation, label=label)


def run(model: str) -> None:
    with TemporaryDirectory(prefix="vexor-batch-") as temporary:
        base = Path(temporary)
        root = base / "documents"
        root.mkdir()
        for name, text in DOCUMENTS.items():
            (root / name).write_text(text + "\n", encoding="utf-8")
        config = {"provider": "local", "model": model, "rerank": "off"}
        for rerank in ("off", "hybrid"):
            config["rerank"] = rerank
            arms: dict[str, list] = {}
            for arm in ("single", "batch"):
                with VexorClient(cache_dir=base / f"{rerank}-{arm}") as client:
                    client.set_config_json(config, replace=True)
                    options = {"path": root, "mode": "full"}
                    client.index(**options)
                    arms[arm] = measure_queries(
                        client, arm, {**options, "include_content": True},
                        f"files/{rerank}/{arm}",
                    )
            assert_equivalent(arms["single"], arms["batch"])

        with VexorClient(cache_dir=base / "memory") as client:
            client.set_config_json(config, replace=True)
            index = client.index_in_memory(path=root, mode="full")
            single = measure(lambda: [index.search(q, include_content=True) for q in QUERIES],
                             label="memory/hybrid/single")
            batch = measure(lambda: index.search_many(QUERIES, include_content=True),
                            label="memory/hybrid/batch")
            assert_equivalent(single, batch)

        for rerank in ("off", "hybrid"):
            arms = {}
            for arm in ("single", "batch"):
                with VexorClient(cache_dir=base / f"records-{rerank}-{arm}") as client:
                    client.set_config_json(config, replace=True)
                    handle = client.collection("documents")
                    handle.upsert_many([
                        {"id": name, "text": text, "metadata": {"tenant": "allowed"}}
                        for name, text in DOCUMENTS.items()
                    ] + [{"id": "private", "text": "private authentication token",
                          "metadata": {"tenant": "other"}}])
                    options = {"filters": {"tenant": "allowed"}, "rerank": rerank}
                    arms[arm] = measure_queries(
                        handle, arm, options, f"collections/{rerank}/{arm}",
                    )
            assert_equivalent(arms["single"], arms["batch"], records=True)
    print("PASS: file, memory, and filtered collection batches match single-query results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    run(parser.parse_args().model)
