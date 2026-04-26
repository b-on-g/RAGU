import numpy as np
import pytest

from ragu.models.sparse_embedder import BM25, BM42


class _FakeFastEmbedSparseEmbedding:
    def __init__(self, indices, values):
        self.indices = np.array(indices)
        self.values = np.array(values)


class _FakeFastEmbedBM25:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        language: str = "english",
        token_max_length: int = 40,
        disable_stemmer: bool = False,
        specific_model_path: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.k = k
        self.b = b
        self.avg_len = avg_len
        self.language = language
        self.token_max_length = token_max_length
        self.disable_stemmer = disable_stemmer
        self.specific_model_path = specific_model_path
        self.kwargs = kwargs
        self.embed_calls = []
        self.query_calls = []

    def embed(self, texts, **kwargs):
        self.embed_calls.append((texts, kwargs))
        return [
            _FakeFastEmbedSparseEmbedding(indices=[idx, idx + 10], values=[1.0, 0.5])
            for idx, _ in enumerate(texts, start=1)
        ]

    def query_embed(self, texts, **kwargs):
        self.query_calls.append((texts, kwargs))
        return [
            _FakeFastEmbedSparseEmbedding(indices=[idx + 100], values=[1.0])
            for idx, _ in enumerate(texts, start=1)
        ]


class _FakeFastEmbedBM42:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers=None,
        alpha: float = 0.5,
        cuda="auto",
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        specific_model_path: str | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.providers = providers
        self.alpha = alpha
        self.cuda = cuda
        self.device_ids = device_ids
        self.lazy_load = lazy_load
        self.specific_model_path = specific_model_path
        self.kwargs = kwargs
        self.embed_calls = []
        self.query_calls = []

    def embed(self, texts, **kwargs):
        self.embed_calls.append((texts, kwargs))
        return [
            _FakeFastEmbedSparseEmbedding(indices=[idx + 200], values=[0.7])
            for idx, _ in enumerate(texts, start=1)
        ]

    def query_embed(self, texts, **kwargs):
        self.query_calls.append((texts, kwargs))
        return [
            _FakeFastEmbedSparseEmbedding(indices=[idx + 300], values=[1.0])
            for idx, _ in enumerate(texts, start=1)
        ]


class _FakeNormalizer:
    def normalize_batch(self, texts: list[str]) -> list[str]:
        return [f"norm::{text}" for text in texts]


def test_bm25_embed_document_uses_normalizer_and_forwards_init_options(monkeypatch):
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM25", _FakeFastEmbedBM25)

    bm25 = BM25(
        model_name="Qdrant/bm25",
        cache_dir="/tmp/cache",
        k=1.7,
        b=0.25,
        avg_len=512.0,
        language="russian",
        token_max_length=64,
        disable_stemmer=True,
        specific_model_path="/tmp/model",
        normalizer=_FakeNormalizer(),
        local_files_only=True,
    )

    embeddings = bm25.embed_document(["hello", "world"])

    assert [embedding.indices for embedding in embeddings] == [[1, 11], [2, 12]]
    assert bm25._model.cache_dir == "/tmp/cache"
    assert bm25._model.k == 1.7
    assert bm25._model.b == 0.25
    assert bm25._model.avg_len == 512.0
    assert bm25._model.language == "russian"
    assert bm25._model.token_max_length == 64
    assert bm25._model.specific_model_path == "/tmp/model"
    assert bm25._model.kwargs == {"local_files_only": True}
    assert bm25._model.embed_calls == [(["norm::hello", "norm::world"], {})]


def test_bm25_embed_query_uses_query_encoder(monkeypatch):
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM25", _FakeFastEmbedBM25)
    bm25 = BM25(disable_stemmer=True, normalizer=_FakeNormalizer())

    embeddings = bm25.embed_query(["question"])

    assert [embedding.indices for embedding in embeddings] == [[101]]
    assert bm25._model.query_calls == [(["norm::question"], {})]


def test_bm25_rejects_custom_normalizer_with_fastembed_stemmer_enabled():
    with pytest.raises(ValueError, match="disable_stemmer"):
        BM25(normalizer=_FakeNormalizer())


def test_bm42_embed_document_forwards_batch_options(monkeypatch):
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM42", _FakeFastEmbedBM42)

    bm42 = BM42(
        cache_dir="/tmp/cache",
        alpha=0.7,
        batch_size=16,
        parallel=3,
        threads=2,
        providers=["CPUExecutionProvider"],
        cuda=False,
        device_ids=[0],
        lazy_load=True,
        specific_model_path="/tmp/bm42",
        local_files_only=True,
    )

    embeddings = bm42.embed_document(["alpha", "beta"])

    assert [embedding.indices for embedding in embeddings] == [[201], [202]]
    assert bm42._model.cache_dir == "/tmp/cache"
    assert bm42._model.alpha == 0.7
    assert bm42._model.threads == 2
    assert bm42._model.providers == ["CPUExecutionProvider"]
    assert bm42._model.cuda is False
    assert bm42._model.device_ids == [0]
    assert bm42._model.lazy_load is True
    assert bm42._model.specific_model_path == "/tmp/bm42"
    assert bm42._model.kwargs == {"local_files_only": True}
    assert bm42._model.embed_calls == [(["alpha", "beta"], {"batch_size": 16, "parallel": 3})]


def test_bm42_embed_query_uses_query_encoder(monkeypatch):
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM42", _FakeFastEmbedBM42)
    bm42 = BM42()

    embeddings = bm42.embed_query(["question"])

    assert [embedding.indices for embedding in embeddings] == [[301]]
    assert bm42._model.query_calls == [(["question"], {})]


def test_bm42_rejects_custom_normalizer():
    with pytest.raises(ValueError, match="does not support a custom normalizer"):
        BM42(normalizer=_FakeNormalizer())


def test_sparse_embedders_return_empty_lists_for_empty_inputs(monkeypatch):
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM25", _FakeFastEmbedBM25)
    monkeypatch.setattr("ragu.models.sparse_embedder.FastEmbedBM42", _FakeFastEmbedBM42)

    bm25 = BM25(disable_stemmer=True, normalizer=_FakeNormalizer())
    bm42 = BM42()

    assert bm25.embed_document([]) == []
    assert bm25.embed_query([]) == []
    assert bm42.embed_document([]) == []
    assert bm42.embed_query([]) == []
