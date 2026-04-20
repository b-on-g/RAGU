# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_storage/vdb_nanovectordb.py

import os
from typing import Any, List
from typing_extensions import override

import numpy as np
from nano_vectordb import NanoVectorDB # pyright: ignore[reportMissingTypeStubs]
from nano_vectordb.dbs import Data

from ragu.common.global_parameters import Settings
from ragu.common.logger import logger
from ragu.storage.base_storage import BaseVectorStorage
from ragu.storage.types import Point, EmbeddingHit


class NanoVectorDBStorage(BaseVectorStorage):
    """
    Vector storage implementation using NanoVectorDB as the backend.

    This class provides a simple vector database for storing and retrieving
    embeddings, enabling similarity search operations such as nearest
    neighbor queries.
    """

    def __init__(
        self,
        embedding_dim: int,
        cosine_threshold: float = 0.2,
        storage_folder: str = Settings.storage_folder,
        filename: str = "data.json",
        **kwargs: Any,
    ):
        """
        Initialize the NanoVectorDB-based vector storage.

        :param embedding_dim: Embedding dimensionality.
        :param cosine_threshold: Minimum cosine similarity threshold for query filtering.
        :param storage_folder: Folder where the vector storage file is located.
        :param filename: Name of the JSON file containing the stored vectors.
        :param kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)

        self.filename = os.path.join(storage_folder, filename)
        self.embedding_dim = embedding_dim
        self.cosine_threshold = cosine_threshold
        self._client = NanoVectorDB(
            embedding_dim,
            storage_file=self.filename
        )

    @override
    async def upsert(self, data: List[Point], **kwargs) -> None:
        """
        Insert or update a batch of embeddings in the database.

        :param data: Embedding records with vectors and metadata.
        :return: List of records successfully inserted or updated.
        """
        if not data:
            logger.warning("Attempted to insert empty data into vector DB.")
            return

        if any([item.sparse_embedding is not None for item in data]):
            logger.warning(f"NanoVDB does not support sparse embeddings. Ignoring.")

        points: List[Data] = []
        for embedding in data:
            item: Data = {
                "__id__": embedding.id,
                "__vector__": np.array(embedding.dense_embedding),
                **embedding.metadata,
            }
            points.append(item)

        if not points:
            return

        self._client.upsert(datas=points)

    @override
    async def query(
            self,
            point: Point,
            **kwargs: Any
    ) -> List[EmbeddingHit]:
        """
        Search for the most similar documents in the vector database.

        Performs a cosine similarity search against all stored vectors,
        returning the top ``k`` results exceeding the similarity threshold.

        :param point: Query embedding payload.
        :param kwargs:
            top_k: Number of nearest neighbors to return.
        :return: List of matched records and their distances.
        """
        if point.dense_embedding is None:
            raise ValueError("Empty dense embedding payload.")

        top_k: int = kwargs.pop("top_k", 20)
        results = self._client.query( # type: ignore
            query=np.array(point.dense_embedding),
            top_k=top_k,
            better_than_threshold=self.cosine_threshold
        )
        hits: List[EmbeddingHit] = []
        for result in results: # type: ignore
            metadata: dict[str, Any] = {
                key: value
                for key, value in result.items() # type: ignore
                if key not in {"__id__", "__metrics__", "__vector__"}
            }
            hits.append(
                EmbeddingHit(
                    id=result["__id__"],
                    distance=float(result["__metrics__"]),
                    metadata=metadata,
                )
            )
        return hits

    async def index_start_callback(self):
        """
        Pre-index hook for interface compatibility.
        """
        pass

    async def query_done_callback(self):
        """
        Post-query hook for interface compatibility.
        """
        pass

    @override
    async def delete(self, ids: List[str], **kwargs: Any) -> None:
        """
        Delete embeddings by their IDs from the vector database.

        :param ids: List of IDs to remove from the vector storage.
        :type ids: List[str]
        """
        if not ids:
            return
        self._client.delete(ids)

    async def index_done_callback(self) -> None:
        """
        Save the current state of the NanoVectorDB to disk.

        This method ensures that any newly inserted or updated vectors
        are persisted in the storage file.
        """
        self._client.save()
