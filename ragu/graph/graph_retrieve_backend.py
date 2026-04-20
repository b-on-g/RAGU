from typing import List

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

    async def query_entities(self, query: str, top_k: int = 20) -> List[Entity]:
        """
        Find entities matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching entities ordered by relevance.
        """
        point = await self.build_query_vectors(query)
        results = await self.knowledge_graph.index.entity_vector_db.query(
            point,
            top_k=top_k,
        )
        entity_ids = [result.id for result in results]
        entities = await self.knowledge_graph.index.get_entities(entity_ids)
        return [entity for entity in entities if entity is not None]

    async def query_relations(self, query: str, top_k: int = 20) -> List[Relation]:
        """
        Find relations matching a free-text query.

        :param query: Search query text.
        :param top_k: Maximum number of results.
        :return: Matching relations ordered by relevance.
        """
        point = await self.build_query_vectors(query)
        results = await self.knowledge_graph.index.relation_vector_db.query(
            point,
            top_k=top_k,
        )
        edge_specs: List[EdgeSpec] = [
            (
                str(result.metadata.get("subject")),
                str(result.metadata.get("object")),
                result.id,
            )
            for result in results
            if result.metadata.get("subject") and result.metadata.get("object")
        ]
        if not edge_specs:
            return []
        relations = await self.knowledge_graph.index.get_relations(edge_specs)
        return [relation for relation in relations if relation is not None]

    async def find_similar_entities(self, entity: Entity, top_k: int = 10) -> List[Entity]:
        """
        Find entities semantically similar to the given entity.

        :param entity: Reference entity to search against.
        :param top_k: Maximum number of results.
        :return: Similar entities ordered by relevance.
        """
        query = f"{entity.entity_name} - {entity.description}"
        return await self.query_entities(query, top_k=top_k)

    async def find_similar_relations(self, relation: Relation, top_k: int = 10) -> List[Relation]:
        """
        Find relations semantically similar to the given relation.

        :param relation: Reference relation to search against.
        :param top_k: Maximum number of results.
        :return: Similar relations ordered by relevance.
        """
        return await self.query_relations(relation.description, top_k=top_k)

    async def query_chunk_hits(self, query: str, top_k: int = 20) -> List[EmbeddingHit]:
        """
        Search chunk vectors and return raw embedding hits.

        :param query: Search query text.
        :param top_k: Maximum number of hits.
        :return: Ranked chunk hits with metadata.
        """
        point = await self.build_query_vectors(query)
        return await self.knowledge_graph.index.chunk_vector_db.query(
            point=point,
            top_k=top_k,
        )

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
