from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, Generic

from pydantic import BaseModel
from ragu.common.base import RaguGenerativeModule
from ragu.models.llm import LLM
from ragu.utils.ragu_utils import always_get_an_event_loop
from ragu.utils.token_truncation import TokenTruncation


ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class SearchEngineRetrieve(ABC, Generic[ResultT]):
    """
    Base response class for retrieval.
    """
    query: str
    result: ResultT
    metrics: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_text(self) -> str:
        """
        How to format the retrieved result.
        """
        ...

    def __str__(self) -> str:
        return self.to_text()


@dataclass(slots=True)
class SearchEngineResponse:
    """
    Default response for search engine.
    """
    query: str
    response: str | BaseModel
    retrieval: SearchEngineRetrieve[Any]
    payload: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if isinstance(self.response, BaseModel):
            return self.response.model_dump_json(indent=4)
        return self.response


class BaseEngine(RaguGenerativeModule, ABC):
    """
    Base interface for RAGU query/search engines.

    Concrete engines implement retrieval (a_search method) and answer generation
    (a_query method) on top of a knowledge graph.
    """

    def __init__(
        self,
        llm: LLM,
        *args: Any,
        max_context_length: int = 30_000,
        tokenizer_backend: Literal["tiktoken", "local"] = "tiktoken",
        tokenizer_model: str = "gpt-4",
        **kwargs: Any,
    ):
        """
        Initialize engine with an LLM client.

        :param client: LLM client.
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

    @abstractmethod
    async def a_search(
        self,
        query,
        *args,
        **kwargs,
    ) -> SearchEngineRetrieve:
        """
        Retrieve context relevant to a query.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        pass

    @abstractmethod
    async def a_query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Execute full query flow and return answer.

        :param query: Input query string.
        :return: Structured search result containing the final answer and retrieval details.
        """
        pass

    async def query(self, query: str, *args, **kwargs) -> SearchEngineResponse:
        """
        Synchronous wrapper for ``a_query``.

        :param query: Input query string.
        :return: Structured search result containing the final answer and retrieval details.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_query(query, *args, **kwargs)
        )

    async def search(
        self,
        query,
        *args,
        **kwargs,
    ) -> SearchEngineRetrieve:
        """
        Synchronous wrapper for ``a_search``.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_search(query, *args, **kwargs)
        )
