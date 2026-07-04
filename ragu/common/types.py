from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourceDocument:
    """
    Original source document stored before chunking.

    :param doc_id: Stable source document identifier referenced by chunks.
    :param content: Raw document text before chunking.
    :param metadata: Optional user/application metadata for the document.
    """
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
