from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from ragu.storage.vdb_storage_adapters.nano_vdb import NanoVectorDBStorage

from tests.storage.qdrant_testkit import load_qdrant_storage


@dataclass(frozen=True)
class VDBBackendCase:
    name: str
    factory: Callable[[Path, pytest.MonkeyPatch], object]
    supports_sparse: bool = False
    supports_persistence: bool = True


def _make_nano_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return NanoVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        cosine_threshold=0.0,
    )


def _make_qdrant_dense_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    return QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
    )


def _make_qdrant_sparse_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QdrantVectorDBStorage = load_qdrant_storage(monkeypatch)
    return QdrantVectorDBStorage(
        embedding_dim=3,
        filename=str(tmp_path / "vdb.json"),
        sparse_type="bm42",
    )


ALL_VDB_CASES = [
    VDBBackendCase(
        name="nano",
        factory=_make_nano_storage,
        supports_sparse=False,
    ),
    VDBBackendCase(
        name="qdrant-dense",
        factory=_make_qdrant_dense_storage,
        supports_sparse=False,
    ),
    VDBBackendCase(
        name="qdrant-bm42",
        factory=_make_qdrant_sparse_storage,
        supports_sparse=True,
    ),
]


def pytest_addoption(parser):
    parser.addoption(
        "--vdb-backend",
        action="append",
        default=[],
        help="Run shared VDB contract tests only for the selected backend ids.",
    )


@pytest.fixture(params=ALL_VDB_CASES, ids=lambda case: case.name)
def vdb_backend_case(request, pytestconfig):
    case = request.param
    selected = pytestconfig.getoption("--vdb-backend")
    if selected and case.name not in selected:
        pytest.skip(f"Backend {case.name} not selected")
    return case


@pytest.fixture
def vdb_storage(vdb_backend_case, tmp_path, monkeypatch):
    return vdb_backend_case.factory(tmp_path, monkeypatch)
