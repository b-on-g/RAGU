"""
ETL: Extract tickets & KB from MSSQL → build Knowledge Graph via RAGU.

50 most informative tickets + KB articles → full GraphRAG pipeline.

Usage:
    docker compose run --rm api python -m etl.extract_and_index
"""

import asyncio
import os
import re
from html.parser import HTMLParser

import pymssql
from dotenv import load_dotenv

from ragu import KnowledgeGraph, SimpleChunker, BuilderArguments, Settings
from ragu.models import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM
from etl.local_embedder import LocalEmbedder


# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    s = _HTMLStripper()
    s.feed(html)
    text = s.get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# MSSQL helpers
# ---------------------------------------------------------------------------

def get_connection():
    return pymssql.connect(
        server=os.getenv("MSSQL_HOST", "mssql"),
        user="SA",
        password=os.getenv("MSSQL_SA_PASSWORD", "BaltBeregHack2026!"),
        database="service_desk_tdbb",
        charset="utf8",
    )


def fetch_lookups(conn) -> dict[str, dict[int, str]]:
    lookups: dict[str, dict[int, str]] = {}
    cursor = conn.cursor(as_dict=True)
    for table in ("Service", "TaskType", "Status", "Priority"):
        cursor.execute(f"SELECT Id, NameXml FROM {table}")
        lookups[table] = {row["Id"]: strip_html(row["NameXml"]) for row in cursor}
    return lookups


def fetch_tickets(conn, lookups: dict[str, dict[int, str]], limit: int = 50) -> list[str]:
    """Fetch most informative tickets — those with longest Comment (richest Q&A)."""
    cursor = conn.cursor(as_dict=True)
    cursor.execute(f"""
        SELECT TOP {limit} Id, Name, Description, Comment,
               ServiceId, TypeId, StatusId, PriorityId
        FROM Task
        WHERE Comment IS NOT NULL AND LEN(Comment) > 100
        ORDER BY LEN(Comment) DESC
    """)

    docs: list[str] = []
    for row in cursor:
        service = lookups["Service"].get(row["ServiceId"], "")
        task_type = lookups["TaskType"].get(row["TypeId"], "")
        status = lookups["Status"].get(row["StatusId"], "")
        priority = lookups["Priority"].get(row["PriorityId"], "")

        desc = strip_html(row["Description"])
        comment = strip_html(row["Comment"])

        parts = [f"Тикет #{row['Id']}: {row['Name'] or ''}"]
        if service:
            parts.append(f"Сервис: {service}")
        if task_type:
            parts.append(f"Тип: {task_type}")
        if status:
            parts.append(f"Статус: {status}")
        if priority:
            parts.append(f"Приоритет: {priority}")
        if desc:
            parts.append(f"Описание: {desc}")
        if comment:
            parts.append(f"Переписка: {comment}")

        docs.append("\n".join(parts))

    return docs


def fetch_kb_articles(conn, limit: int = 50) -> list[str]:
    cursor = conn.cursor(as_dict=True)
    cursor.execute(f"""
        SELECT TOP {limit} Id, Name, Description
        FROM KBDocument
        WHERE IsPublished = 1 AND Description IS NOT NULL AND LEN(Description) > 50
        ORDER BY Rating DESC, Id DESC
    """)

    docs: list[str] = []
    for row in cursor:
        desc = strip_html(row["Description"])
        parts = [f"Статья базы знаний #{row['Id']}: {row['Name'] or ''}"]
        if desc:
            parts.append(desc)
        docs.append("\n".join(parts))

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()

    storage = os.getenv("RAGU_STORAGE", "ragu_data")
    Settings.storage_folder = storage
    Settings.language = "russian"

    llm_client = CachedAsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://models.github.ai/inference"),
        api_key=os.getenv("LLM_API_KEY", ""),
        rate_max_per_minute=5,
        rate_max_simultaneous=1,
        rate_min_delay=8,
        retry_times_sec=(30, 60, 120, 240),
    )
    llm = LLMOpenAI(client=llm_client, model_name=os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"))

    print("[ETL] Loading local embedder...")
    embedder = LocalEmbedder("intfloat/multilingual-e5-small")

    # Extract from MSSQL
    print("[ETL] Connecting to MSSQL...")
    conn = get_connection()

    print("[ETL] Loading lookups...")
    lookups = fetch_lookups(conn)
    for table, data in lookups.items():
        print(f"  {table}: {len(data)} entries")

    print("[ETL] Extracting top 10 KB articles (by rating)...")
    kb_docs = fetch_kb_articles(conn, limit=10)
    print(f"  {len(kb_docs)} KB articles")

    print("[ETL] Extracting top 10 tickets (richest Q&A threads)...")
    ticket_docs = fetch_tickets(conn, lookups, limit=10)
    print(f"  {len(ticket_docs)} tickets")

    conn.close()

    all_docs = kb_docs + ticket_docs
    print(f"[ETL] Total: {len(all_docs)} documents")

    # Build full Knowledge Graph (entities, relations, communities, summaries)
    print("[ETL] Building Knowledge Graph...")
    extractor = ArtifactsExtractorLLM(llm=llm, do_validation=False)

    kg = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(max_chunk_size=1000),
        artifact_extractor=extractor,
        builder_settings=BuilderArguments(
            use_llm_summarization=True,
            vectorize_chunks=True,
        ),
    )

    await kg.build_from_docs(all_docs)

    print(f"[ETL] Done! Knowledge Graph saved to {storage}/")


if __name__ == "__main__":
    asyncio.run(main())
