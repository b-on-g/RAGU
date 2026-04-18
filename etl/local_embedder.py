"""Local sentence-transformers embedder implementing RAGU's Embedder interface."""

from typing import Any
from ragu.models.embedder import Embedder, FLOATS


class LocalEmbedder(Embedder):
    """Embedder using sentence-transformers locally. No API calls, no rate limits."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"[LocalEmbedder] Loaded {model_name}, dim={self._dim}")

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]] | FLOATS:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=bool(desc),
            batch_size=64,
        )
        return embeddings.tolist()
