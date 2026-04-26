# Tests

The test suite is pytest-based and is organized by package area. The default
pytest configuration discovers tests under `tests/` only; root-level ad hoc
scripts such as `test.py` are not part of the normal suite.

## Structure

```text
tests/
├── conftest.py                                  # Shared text fixtures and event loop setup
├── chunker/
│   └── test_chunkers.py                         # Simple, semantic, and smart chunker behavior
├── common/
│   ├── test_batch_generator.py                  # Batch iteration utility
│   ├── test_cache.py                            # Disk/cache helpers
│   └── test_env.py                              # Environment and settings resolution
├── embedder/
│   ├── test_embedders.py                        # Dense embedder wrappers
│   └── test_sparse_embedders.py                 # BM25/BM42 sparse embedder wrappers
├── graph/
│   ├── test_artifacts_summarizer.py             # Entity/relation summarization and deduplication
│   ├── test_builder_modules.py                  # Graph builder modules
│   ├── test_community_summarizer.py             # Community report generation
│   ├── test_graph_loading.py                    # Loading a serialized graph from storage
│   ├── test_graph_types.py                      # Entity, relation, and community dataclasses
│   ├── test_index_crud.py                       # Index CRUD, cascades, vectors, and consistency checks
│   ├── test_knowledge_graph_merge.py            # KnowledgeGraph high-level merge/update behavior
│   └── test_merge_logic.py                      # Pure entity/relation merge helpers
├── llm/
│   └── test_cached_openai.py                    # Cached OpenAI client against a local mock server
├── rerank/
│   ├── test_api_rerankers.py                    # API-based reranker wrappers
│   ├── test_base_reranker.py                    # Base reranker behavior
│   └── test_local_rerankers.py                  # Local reranker batching and ordering
├── search_engine/
│   ├── conftest.py                              # Real KnowledgeGraph fixture backed by kg_for_test/
│   ├── test_global_search_engine.py             # Global search behavior
│   ├── test_local_search_engine.py              # Local search behavior
│   ├── test_mix_search_engine.py                # Multi-engine orchestration
│   └── test_naive_search_engine.py              # Chunk/vector based search
├── storage/
│   ├── conftest.py                              # Shared vector DB backend contract cases
│   ├── qdrant_testkit.py                        # In-memory fake Qdrant implementation
│   ├── test_backend_batch_operations.py         # Batch graph/KV/vector storage operations
│   ├── test_json_storage.py                     # JSON KV storage adapter
│   ├── test_memgraph_adapter.py                 # Optional Memgraph integration tests
│   ├── test_networkx_adapter.py                 # NetworkX graph storage adapter
│   ├── test_qdrant_vdb_sparse_modes.py          # Qdrant sparse/hybrid configuration behavior
│   ├── test_qdrant_vdb_storage.py               # Qdrant vector storage edge cases
│   └── test_vdb_contract.py                     # Shared VDB contract across Nano/Qdrant backends
├── triplet/
│   └── test_llm_exception_handling.py           # LLM extractor fail-fast behavior
├── utils/
│   ├── test_ragu_utils.py                       # Hash IDs, async context helpers, file readers
│   └── test_token_truncation.py                 # Token-aware text truncation
└── kg_for_test/                                 # Serialized graph fixture for integration-level tests
```

## Fixtures And State

- Tests that need temporary RAGU storage should use `monkeypatch.setattr(Settings, "storage_folder", ...)` so global state is restored after each test.
- `tests/search_engine/conftest.py` loads `tests/kg_for_test/` into a real `KnowledgeGraph` for search-engine integration-style tests.
- `tests/storage/conftest.py` parametrizes shared vector DB contract tests across NanoVDB and fake-Qdrant backends.
- Memgraph tests are marked `integration` and skip unless `MEMGRAPH_URI` and `MEMGRAPH_PASSWORD` are configured.

## Useful Commands

```bash
pytest -q --no-cov
pytest -q --no-cov -m "not slow and not integration"
pytest -q --cov-report=term-missing:skip-covered
```
