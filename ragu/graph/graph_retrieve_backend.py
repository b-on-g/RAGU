from typing import List, Tuple

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation
from ragu.models.embedder import Embedder
from ragu.models.scorer import Scorer
from ragu.models.sparse_embedder import SparseEmbedder
from ragu.storage.base_storage import EdgeSpec
from ragu.storage.types import Point, EmbeddingHit, SparseEmbedding, DenseEmbedding

from ragu.graph.knowledge_graph import KnowledgeGraph


class GraphRetriever:
    """
    Query-time retrieval helper for graph vector search.
    """
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder | None = None,
        reranker: Scorer | None = None,
    ) -> None:
        """
        Initialize a retriever bound to an existing knowledge graph.

        :param knowledge_graph: Graph container exposing storage backends.
        :param embedder: Dense embedder used for query encoding.
        :param sparse_embedder: Optional sparse embedder for hybrid retrieval.
        :param reranker: Optional reranker reserved for post-retrieval scoring.
        """
        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker

    async def query_entities(self, query: str, top_k: int = 20) -> Tuple[List[Entity], List[EmbeddingHit]]:
        """
        Find entities matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching entities ordered by relevance and their aligned vector hits.
        """
        point = await self.build_query_vectors(query)
        results = await self.knowledge_graph.index.nodes_vector_db.query(
            point,
            top_k=top_k,
        )
        entity_ids = [result.id for result in results]
        entities = await self.knowledge_graph.index.get_nodes(entity_ids)
        filtered_entities: List[Entity] = []
        filtered_hits: List[EmbeddingHit] = []
        for hit, entity in zip(results, entities):
            if entity is None:
                continue
            filtered_entities.append(entity)
            filtered_hits.append(hit)
        return filtered_entities, filtered_hits

    async def query_relations(self, query: str, top_k: int = 20) -> Tuple[List[Relation], List[EmbeddingHit]]:
        """
        Find relations matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching relations ordered by relevance and their aligned vector hits.
        """
        point = await self.build_query_vectors(query)
        results = await self.knowledge_graph.index.edges_vector_db.query(
            point,
            top_k=top_k,
        )
        edge_specs: List[EdgeSpec] = []
        filtered_hits: List[EmbeddingHit] = []
        for result in results:
            subject_id = result.metadata.get("subject_id")
            object_id = result.metadata.get("object_id")
            if not subject_id or not object_id:
                continue
            edge_specs.append(
                (
                    str(subject_id),
                    str(object_id),
                    result.id,
                )
            )
            filtered_hits.append(result)
        if not edge_specs:
            return [], []
        relations = await self.knowledge_graph.index.get_edges(edge_specs)
        filtered_relations: List[Relation] = []
        aligned_hits: List[EmbeddingHit] = []
        for hit, relation in zip(filtered_hits, relations):
            if relation is None:
                continue
            filtered_relations.append(relation)
            aligned_hits.append(hit)
        return filtered_relations, aligned_hits

    async def query_chunks(self, query: str, top_k: int = 20) -> Tuple[List[Chunk], List[EmbeddingHit]]:
        """
        Search chunk vectors and return resolved chunks with aligned embedding hits.

        :param query: Search query text.
        :param top_k: Maximum number of hits.
        :return: Ranked chunks with aligned vector hits.
        """
        point = await self.build_query_vectors(query)
        results = await self.knowledge_graph.index.chunks_vector_db.query(
            point=point,
            top_k=top_k,
        )
        chunk_ids = [result.id for result in results]
        chunk_data_list = await self.knowledge_graph.index.chunks_kv_storage.get_by_ids(chunk_ids)

        chunks: List[Chunk] = []
        filtered_hits: List[EmbeddingHit] = []
        for chunk_id, chunk_data, hit in zip(chunk_ids, chunk_data_list, results):
            if chunk_data is None:
                continue
            chunk = Chunk(
                content=chunk_data.get("content", ""),
                chunk_order_idx=chunk_data.get("chunk_order_idx", 0),
                doc_id=chunk_data.get("doc_id", ""),
                num_tokens=chunk_data.get("num_tokens"),
            )
            setattr(chunk, "id", chunk_id)
            chunks.append(chunk)
            filtered_hits.append(hit)
        return chunks, filtered_hits

    async def find_similar_entities(
        self,
        entity: Entity,
        top_k: int = 10,
    ) -> Tuple[List[Entity], List[EmbeddingHit]]:
        """
        Find entities semantically similar to the given entity.

        :param entity: Reference entity to search against.
        :param top_k: Maximum number of results.
        :return: Similar entities ordered by relevance and their aligned vector hits.
        """
        query = f"{entity.entity_name} - {entity.description}"
        return await self.query_entities(query, top_k=top_k)

    async def find_similar_relations(
        self,
        relation: Relation,
        top_k: int = 10,
    ) -> Tuple[List[Relation], List[EmbeddingHit]]:
        """
        Find relations semantically similar to the given relation.

        :param relation: Reference relation to search against.
        :param top_k: Maximum number of results.
        :return: Similar relations ordered by relevance and their aligned vector hits.
        """
        return await self.query_relations(relation.description, top_k=top_k)

    async def build_query_vectors(self, query: str) -> Point:
        """
        Encode a query into dense and optional sparse vectors.

        :param query: Search query text.
        :return: Point carrying query-time vector payloads.
        """
        dense_query: DenseEmbedding = await self.embedder.embed_text(query) # type: ignore
        sparse_query: SparseEmbedding | None = None
        if self.sparse_embedder is not None:
            sparse_vectors = self.sparse_embedder.embed_query([query])
            if len(sparse_vectors) != 1:
                raise ValueError("Sparse query embedder must return exactly one vector for one query")
            sparse_query = sparse_vectors[0]
        return Point(dense_embedding=dense_query, sparse_embedding=sparse_query)
