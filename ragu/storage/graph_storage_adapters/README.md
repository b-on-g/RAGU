# Module: ragu.storage.graph_storage_adapters

## Role in RAGU Pipeline

This package provides graph backend implementations for `Index`. Graph storage persists entities and relations after extraction and supports local-search neighborhood traversal.

Pipeline position:

```text
Entity/Relation -> BaseGraphStorage adapter -> graph backend -> LocalSearchEngine
```

## Overview

Graph adapters implement `BaseGraphStorage` for different backends while preserving RAGU's directed multigraph contract.

## Key Components

### NetworkXStorage

- Purpose: default local graph backend.
- Persistence: reads and writes GML files.
- Important parameters: `filename`, `node_cls`, `edge_cls`.

## Data Flow

Input: `Entity` nodes and `Relation` edges.

Output: stored nodes, stored multigraph edges, edge-degree values, incident edge lists.

Used by:

- `ragu.graph.Index`
- `ragu.graph.KnowledgeGraph`
- `ragu.search_engine.LocalSearchEngine`

## Usage Examples

### Example 1 - Minimal usage

```python
import asyncio

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


async def main():
    storage = NetworkXStorage(
        filename="knowledge_graph.gml",
        node_cls=Entity,
        edge_cls=Relation,
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        subject_id=guido.id,
        object_id=python.id,
        subject_name=guido.entity_name,
        object_name=python.entity_name,
        relation_type="CREATED",
        description="Guido van Rossum created Python.",
        source_chunk_id=["chunk-1"],
    )

    await storage.upsert_nodes([python, guido])
    await storage.upsert_edges([relation])

    print(await storage.get_nodes([python.id, guido.id]))
    print(await storage.get_edges([(guido.id, python.id, relation.id)]))
    print(await storage.get_all_edges_for_nodes([python.id]))


asyncio.run(main())
```

### Example 2 - Delete graph records

```python
import asyncio

from ragu.graph.types import Entity, Relation
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


async def main():
    storage = NetworkXStorage(
        filename="knowledge_graph.gml",
        node_cls=Entity,
        edge_cls=Relation,
    )
    python = Entity("Python", "Language", "A programming language.", ["chunk-1"])
    guido = Entity("Guido van Rossum", "Person", "Creator of Python.", ["chunk-1"])
    relation = Relation(
        guido.id,
        python.id,
        guido.entity_name,
        python.entity_name,
        "CREATED",
        "Guido van Rossum created Python.",
    )

    await storage.upsert_nodes([python, guido])
    await storage.upsert_edges([relation])
    await storage.delete_edges([(guido.id, python.id, relation.id)])
    await storage.delete_nodes([python.id])

    print(await storage.get_nodes([python.id, guido.id]))


asyncio.run(main())
```

### Example 3 - Pipeline usage

```python
from ragu import BuilderArguments, KnowledgeGraph, StorageArguments
from ragu.models.embedder import EmbedderOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage


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
    storage_settings=StorageArguments(graph_backend_storage=NetworkXStorage),
)
```

## Integration Points

- `Index` calls graph adapters for CRUD and cascade deletion.
- `GraphRetriever` resolves relation vector hits through graph edge specs.
- Local search uses graph edges around retrieved entities.

## Configuration

`StorageArguments.graph_storage_kwargs` is merged with the default `knowledge_graph.gml` filename under `Settings.storage_folder`.

## Dependencies

- `networkx`

## Notes / Pitfalls

- RAGU edge identity is `(subject_id, object_id, relation_id)`.
- `NetworkXStorage.index_done_callback()` writes GML to disk.
- Deleting a node removes connected graph edges.
