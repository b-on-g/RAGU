import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ragu.common.global_parameters import Settings
from ragu.common.types import SourceDocument
from ragu.graph.index import Index, StorageArguments
from ragu.graph.graph_builder_pipeline import BuilderArguments
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder


class DummyEmbedder(Embedder):
    @property
    def dim(self) -> int:
        return 3

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        return [0.1, 0.2, 0.3]


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    embedder = AsyncMock(spec=Embedder)
    embedder.dim = 3
    embedder.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embedder.batch_embed_text = AsyncMock(
        side_effect=lambda texts, **kwargs: [[0.1, 0.2, 0.3] for _ in texts]
    )
    return Index(embedder=embedder, arguments=StorageArguments())


async def test_index_upserts_and_gets_source_documents(index):
    documents = [
        SourceDocument(doc_id="doc-1", content="Raw document one", metadata={"path": "one.txt"}),
        SourceDocument(doc_id="doc-2", content="Raw document two"),
    ]

    await index.upsert_documents(documents)

    retrieved = await index.get_documents_by_ids(["doc-1", "missing", "doc-2"])

    assert retrieved[0] == documents[0]
    assert retrieved[1] is None
    assert retrieved[2] == documents[1]


async def test_build_from_docs_stores_raw_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "storage_folder", str(tmp_path / "storage"))
    docs = ["Alpha raw text", "Beta raw text"]
    graph = KnowledgeGraph(
        llm=None,
        embedder=DummyEmbedder(),
        builder_settings=BuilderArguments(
            build_only_vector_context=True,
            make_community_summary=False,
        ),
    )

    await graph.build_from_docs(docs)

    retrieved = await graph.get_documents_by_ids(["doc_0", "doc_1"])
    assert [document.content for document in retrieved if document is not None] == docs

    stored = json.loads(Path(graph.index.documents_kv_storage.filename).read_text(encoding="utf-8"))
    assert stored["doc_0"]["content"] == "Alpha raw text"
    assert stored["doc_1"]["content"] == "Beta raw text"
