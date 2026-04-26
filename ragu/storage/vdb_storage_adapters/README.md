# Module: ragu.storage.vdb_storage_adapters

## Role in RAGU Pipeline

This package stores dense and optional sparse vectors for entities, relations, and chunks. Search engines depend on these adapters for similarity retrieval.

Pipeline position:

```text
text -> Embedder/SparseEmbedder -> Point -> vector DB -> EmbeddingHit
```

## Overview

Vector adapters implement `BaseVectorStorage`. RAGU ships a lightweight local adapter and a Qdrant adapter for production-style dense and hybrid retrieval.

## Key Components

### NanoVectorDBStorage

- Purpose: default local dense vector storage.
- Used for: development, tests, small examples.

### QdrantVectorDBStorage

- Purpose: Qdrant-backed dense-only or hybrid dense+sparse storage.
- Important parameters: `embedding_dim`, `collection_name`, `location`, `url`, `sparse_type`.
- Sparse modes: `bm25`, `bm42`, `splade`, `custom`.

## Data Flow

Input: `Point` objects with dense and/or sparse vectors.

Output: ranked `EmbeddingHit` results.

Used by:

- `Index.nodes_vector_db`
- `Index.edges_vector_db`
- `Index.chunks_vector_db`
- `GraphRetriever`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio
import numpy as np

from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


async def main():
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        location=":memory:",
        collection_name="example_vectors",
    )
    await storage.upsert([
        Point(id="entity-1", dense_embedding=np.array([1.0, 0.0, 0.0])),
    ])
    hits = await storage.query(
        Point(id="query", dense_embedding=np.array([1.0, 0.0, 0.0])),
        top_k=1,
    )
    print(hits[0].id)


asyncio.run(main())
```

### Example 2 - Point lookup and delete

```python
import asyncio
import numpy as np

from ragu.storage.types import Point
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


async def main():
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        location=":memory:",
        collection_name="point_crud_vectors",
    )
    await storage.upsert([
        Point(
            id="entity-1",
            dense_embedding=np.array([1.0, 0.0, 0.0]),
            metadata={"entity_name": "Python"},
        )
    ])

    print(await storage.get_points_by_ids(["entity-1"]))
    print(await storage.get_payloads_by_ids(["entity-1"]))

    await storage.delete(["entity-1"])
    print(await storage.get_points_by_ids(["entity-1"]))


asyncio.run(main())
```

### Example 3 - Hybrid dense + sparse point

```python
import asyncio
import numpy as np

from ragu.storage.types import Point, SparseEmbedding
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


async def main():
    storage = QdrantVectorDBStorage(
        embedding_dim=3,
        location=":memory:",
        collection_name="hybrid_vectors",
        sparse_type="bm25",
    )
    await storage.upsert([
        Point(
            id="chunk-1",
            dense_embedding=np.array([0.1, 0.2, 0.3]),
            sparse_embedding=SparseEmbedding(indices=[10, 25], values=[0.8, 0.4]),
            metadata={"content": "Hybrid retrieval uses dense and sparse vectors."},
        )
    ])

    hits = await storage.query(
        Point(
            dense_embedding=np.array([0.1, 0.2, 0.3]),
            sparse_embedding=SparseEmbedding(indices=[10], values=[0.8]),
        ),
        top_k=1,
    )
    print(hits)


asyncio.run(main())
```

### Example 4 - Pipeline usage

```python
from ragu import BuilderArguments, KnowledgeGraph, StorageArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.vdb_storage_adapters.qdrant_vdb import QdrantVectorDBStorage


client = CachedAsyncOpenAI(
    base_url="https://api.openai.com/v1",
    api_key="dummy-api-token",
)
embedder = EmbedderOpenAI(
    client=client,
    model_name="text-embedding-3-small",
    dim=1536,
)

graph = KnowledgeGraph(
    llm=None,
    embedder=embedder,
    builder_settings=BuilderArguments(build_only_vector_context=True),
    storage_settings=StorageArguments(
        vdb_storage_type=QdrantVectorDBStorage,
        vdb_storage_kwargs={"location": ":memory:"},
    ),
)
```

## Integration Points

- `Index` writes vectors for graph artifacts and chunks.
- `GraphRetriever` creates query `Point` objects and calls `query()`.
- Sparse embedders must be paired with a vector adapter configured for sparse vectors.

## Configuration

Qdrant modes:

- dense only: `sparse_type=None`
- BM25 hybrid: `sparse_type="bm25"`
- BM42 hybrid: `sparse_type="bm42"`
- SPLADE hybrid: `sparse_type="splade"`

## Dependencies

- `numpy`
- `qdrant-client`
- local NanoVectorDB dependencies

## Notes / Pitfalls

- Dense vector dimensions must match `Embedder.dim`.
- Existing Qdrant collections are schema-validated on first use.
- Qdrant hybrid queries use reciprocal-rank fusion.
