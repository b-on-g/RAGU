import asyncio
from typing import Any

from fastembed import TextEmbedding
from ragu import (
    BuilderArguments,
    KnowledgeGraph,
    NaiveSearchEngine,
    Settings,
    SimpleChunker,
)
from ragu.models.llm import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.utils.ragu_utils import read_text_from_files

from openai import AsyncOpenAI


class FastEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim: int = 384,
    ) -> None:
        self.model = TextEmbedding(model_name=model_name)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        vectors = await self.batch_embed_text([text], **kwargs)
        return vectors[0]

    async def batch_embed_text(
        self,
        texts: list[str],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_batch, texts)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]


async def main():
    # Configure working directory and language
    Settings.storage_folder = "ragu_working_dir/example_naive_index"
    Settings.language = "russian"
    Settings.embedder_token_limit = 6000

    # Load documents
    docs = read_text_from_files("data")

    # Initialize chunker
    chunker = SimpleChunker(max_chunk_size=1000)

    YANDEX_FOLDER_ID=None
    YANDEX_API_KEY=None
    YANDEX_BASE_URL='https://ai.api.cloud.yandex.net/v1'

    yandex_client = AsyncOpenAI(
        api_key=YANDEX_API_KEY,
        project=YANDEX_FOLDER_ID,
        base_url=YANDEX_BASE_URL,
        max_retries=0,
    )

    # Set up shared OpenAI client with rate limiting
    client = CachedAsyncOpenAI(
        client=yandex_client,
        rate_min_delay=1.5,
        rate_max_simultaneous=1,
        rate_max_per_minute=30,
        retry_times_sec=(5, 10, 20),
        cache="ragu_working_dir/openai_cache",
        embed_timeout=120.0,
    )

    YANDEX_LLM_MODEL = "yandexgpt-5-pro"

    llm_model_name = f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_LLM_MODEL}"

    llm = LLMOpenAI(
        client=client,
        model_name=llm_model_name,
    )

    embedder = FastEmbedder()

    # Configure chunk-only vector index for fast RAG over the source document.
    builder_settings = BuilderArguments(
        build_only_vector_context=True,
        use_llm_summarization=False,
        make_community_summary=False,
    )

    # Build knowledge graph
    knowledge_graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=chunker,
        builder_settings=builder_settings,
    )
    await knowledge_graph.build_from_docs(docs)

    # Set up search engine
    search_engine = NaiveSearchEngine(
        llm,
        knowledge_graph,
        embedder,
    )

    # Run local search
    questions = [
        "При содействии кого выпускался журнал?",
        "Когда введено в эксплуатацию здание вентялиционного ствола номер два", # 1991
    ]

    for question in questions:
        print(f'\nВопрос: {question}')
        answer = await search_engine.a_query(question)
        print(f'Ответ: {answer.response}')


if __name__ == "__main__":
    asyncio.run(main())
