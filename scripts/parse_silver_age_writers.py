from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


API_URL = "https://ru.wikipedia.org/w/api.php"
DEFAULT_OUTPUT = "data/silver_age"
DEFAULT_DELAY = 0.2
MAX_CATEGORY_DEPTH = 4
SEARCH_LIMIT_PER_QUERY = 50

CATEGORY_SEEDS: tuple[str, ...] = (
    "Категория:Русские поэты Серебряного века",
)

SEARCH_QUERIES: tuple[str, ...] = (
    '"русский писатель" "Серебряного века"',
    '"русская писательница" "Серебряного века"',
    '"русский поэт" "Серебряного века"',
    '"русская поэтесса" "Серебряного века"',
    '"русский драматург" "Серебряного века"',
    '"русский прозаик" "Серебряного века"',
)

LIST_PAGE_SEEDS: tuple[str, ...] = (
    "Серебряный век русской поэзии",
    "Русские поэты-футуристы",
    "Русский футуризм",
    "Кубофутуризм",
    "Эгофутуризм",
    "Русский символизм",
    "Младосимволисты",
    "Акмеизм",
    "Имажинизм",
    "Новокрестьянские поэты",
)

RELEVANT_LINK_SECTIONS = frozenset(
    {
        "авторы",
        "список авторов",
        "представители",
        "основные представители",
        "персоналии",
        "поэты",
        "поэты-футуристы",
        "символисты",
        "акмеисты",
        "футуристы",
        "имажинисты",
    }
)

NOISE_TAIL_SECTIONS = frozenset(
    {
        "см. также",
        "смотри также",
        "примечания",
        "комментарии",
        "литература",
        "ссылки",
        "внешние ссылки",
    }
)

WRITER_TERMS = (
    "писател",
    "поэт",
    "поэтесс",
    "прозаик",
    "драматург",
)

SILVER_AGE_MOVEMENT_TERMS = (
    "русский футуризм",
    "кубофутуризм",
    "эгофутуризм",
    "русский символизм",
    "младосимвол",
    "акмеизм",
    "имажинизм",
    "новокрестьянск",
    "русский авангард",
)

PERSON_CATEGORY_TERMS = (
    "персоналии",
    "родившиеся",
    "умершие",
)

NON_PERSON_TITLE_MARKERS = (
    "(журнал",
    "(альманах",
    "(сборник",
    "(поэма",
    "(стихотворение",
    "(роман",
    "(кафе",
    "(издательство",
)


@dataclass(slots=True)
class Candidate:
    pageid: int
    ns: int
    title: str
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PageRecord:
    pageid: int
    ns: int
    title: str
    extract: str
    fullurl: str
    lastrevid: int | None
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FilterDecision:
    included: bool
    reason: str


@dataclass(frozen=True, slots=True)
class IncludedPage:
    page: PageRecord
    filename: str
    reason: str
    sources: tuple[str, ...]
    markdown: str


@dataclass(frozen=True, slots=True)
class RejectedPage:
    pageid: int | None
    title: str
    reason: str
    sources: tuple[str, ...]


class WikiClient:
    def __init__(self, delay: float = DEFAULT_DELAY) -> None:
        import requests

        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "RAGU Silver Age writers parser/1.0 "
                    "(https://github.com/AsphodelRem/RAGU)"
                )
            }
        )

    def get(self, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error']}")
        if self.delay > 0:
            time.sleep(self.delay)
        return data


def _normalize(value: str) -> str:
    return value.casefold().replace("ё", "е")


def _contains_any(value: str, terms: Iterable[str]) -> bool:
    normalized = _normalize(value)
    return any(term in normalized for term in terms)


def _normalize_section_title(value: str) -> str:
    normalized = _normalize(value).strip()
    return re.sub(r"\s+", " ", normalized).strip(" .:;")


def _parse_wiki_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", line.strip())
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def _chunked(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def safe_markdown_filename(title: str, max_length: int = 120) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", title)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if not stem:
        stem = "page"
    if len(stem) > max_length:
        stem = stem[:max_length].rstrip(" .")
    return f"{stem}.md"


def strip_noise_tail_sections(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        heading = _parse_wiki_heading(line)
        if not heading:
            continue
        level, title = heading
        if level == 2 and _normalize_section_title(title) in NOISE_TAIL_SECTIONS:
            return "\n".join(lines[:index]).rstrip()
    return text.rstrip()


def convert_wiki_headings_to_markdown(text: str) -> str:
    converted: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        heading = _parse_wiki_heading(line)
        if heading:
            level, title = heading
            converted.append(f"{'#' * min(level, 6)} {title}")
        else:
            converted.append(line.rstrip())
    return "\n".join(converted).strip()


def normalize_markdown_body(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            normalized.append(line)
            continue
        blank_count += 1
        if blank_count <= 2:
            normalized.append("")
    return "\n".join(normalized).strip()


def build_markdown(page: PageRecord, generated_at: str) -> str:
    body = strip_noise_tail_sections(page.extract)
    body = convert_wiki_headings_to_markdown(body)
    body = normalize_markdown_body(body)
    revision = page.lastrevid if page.lastrevid is not None else ""

    parts = [
        f"# {page.title}",
        "",
        f"Источник: {page.fullurl}",
        f"Page ID: {page.pageid}",
        f"Revision: {revision}",
        f"Дата выгрузки: {generated_at}",
        "",
    ]
    if body:
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


def classify_page_for_inclusion(
    *,
    title: str,
    ns: int,
    categories: Sequence[str],
    extract: str,
    sources: Sequence[str] = (),
) -> FilterDecision:
    if ns != 0:
        return FilterDecision(False, "unsupported_namespace")

    normalized_title = _normalize(title)
    if normalized_title.startswith("список "):
        return FilterDecision(False, "list_page")

    category_text = "\n".join(categories)
    intro = "\n\n".join(extract.split("\n\n")[:2])
    source_text = "\n".join(sources)
    combined = "\n".join((category_text, intro, source_text))
    page_text = "\n".join((category_text, intro))

    silver_age_signal = _contains_any(
        combined,
        ("серебряного века", "серебряный век"),
    ) or _contains_any(
        page_text,
        SILVER_AGE_MOVEMENT_TERMS,
    )
    writer_signal = _contains_any(page_text, WRITER_TERMS)
    person_signal = _contains_any(category_text, PERSON_CATEGORY_TERMS) or _contains_any(
        intro,
        ("родился", "родилась", "род.", "урожд"),
    )

    if _contains_any(title, NON_PERSON_TITLE_MARKERS) and not person_signal:
        return FilterDecision(False, "non_person_title")
    if not silver_age_signal:
        return FilterDecision(False, "no_silver_age_signal")
    if not writer_signal:
        return FilterDecision(False, "no_writer_signal")
    if not person_signal:
        return FilterDecision(False, "no_person_signal")

    if any(source.startswith("seed_category:") for source in sources):
        return FilterDecision(True, "seed_category_writer")
    if _contains_any(category_text, ("серебряного века", "серебряный век")):
        return FilterDecision(True, "category_silver_age_writer")
    if any(source.startswith("list_page:") for source in sources):
        return FilterDecision(True, "list_page_movement_writer")
    return FilterDecision(True, "search_silver_age_writer")


def add_candidate(
    candidates: dict[int, Candidate],
    *,
    pageid: int,
    ns: int,
    title: str,
    sources: Iterable[str],
) -> None:
    candidate = candidates.get(pageid)
    if candidate is None:
        candidate = Candidate(pageid=pageid, ns=ns, title=title)
        candidates[pageid] = candidate
    candidate.sources.update(sources)


def collect_category_candidates(
    client: WikiClient,
    candidates: dict[int, Candidate],
    *,
    root_categories: Sequence[str] = CATEGORY_SEEDS,
) -> None:
    visited_categories: set[str] = set()
    for root_category in root_categories:
        _collect_category_members(
            client,
            candidates,
            category=root_category,
            root_category=root_category,
            visited_categories=visited_categories,
            depth=0,
        )


def _collect_category_members(
    client: WikiClient,
    candidates: dict[int, Candidate],
    *,
    category: str,
    root_category: str,
    visited_categories: set[str],
    depth: int,
) -> None:
    if depth > MAX_CATEGORY_DEPTH or category in visited_categories:
        return
    visited_categories.add(category)

    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmnamespace": "0|14",
        "cmlimit": "max",
        "format": "json",
        "formatversion": "2",
    }

    while True:
        data = client.get(params)
        members = data.get("query", {}).get("categorymembers", [])
        for member in members:
            ns = int(member.get("ns", -1))
            title = str(member.get("title", ""))
            if ns == 0:
                add_candidate(
                    candidates,
                    pageid=int(member["pageid"]),
                    ns=ns,
                    title=title,
                    sources=(
                        f"category:{category}",
                        f"seed_category:{root_category}",
                    ),
                )
            elif ns == 14:
                _collect_category_members(
                    client,
                    candidates,
                    category=title,
                    root_category=root_category,
                    visited_categories=visited_categories,
                    depth=depth + 1,
                )

        continuation = data.get("continue")
        if not continuation:
            break
        params.update(continuation)


def collect_search_candidates(
    client: WikiClient,
    candidates: dict[int, Candidate],
    *,
    queries: Sequence[str] = SEARCH_QUERIES,
) -> None:
    for query in queries:
        data = client.get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": SEARCH_LIMIT_PER_QUERY,
                "format": "json",
                "formatversion": "2",
            }
        )
        for item in data.get("query", {}).get("search", []):
            add_candidate(
                candidates,
                pageid=int(item["pageid"]),
                ns=int(item.get("ns", 0)),
                title=str(item["title"]),
                sources=(f"search:{query}",),
            )


def collect_list_page_candidates(
    client: WikiClient,
    candidates: dict[int, Candidate],
    *,
    pages: Sequence[str] = LIST_PAGE_SEEDS,
) -> None:
    for page_title in pages:
        links = extract_relevant_links(fetch_wikitext(client, page_title))
        resolved_pages = resolve_titles(client, sorted(links))
        for page in resolved_pages:
            add_candidate(
                candidates,
                pageid=page["pageid"],
                ns=page["ns"],
                title=page["title"],
                sources=(f"list_page:{page_title}",),
            )


def fetch_wikitext(client: WikiClient, page_title: str) -> str:
    data = client.get(
        {
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "redirects": 1,
            "format": "json",
            "formatversion": "2",
        }
    )
    parse_data = data.get("parse", {})
    wikitext = parse_data.get("wikitext", "")
    if isinstance(wikitext, dict):
        return str(wikitext.get("*", ""))
    return str(wikitext)


def extract_relevant_links(wikitext: str) -> set[str]:
    relevant_sections = _extract_relevant_sections(wikitext)
    text = "\n".join(relevant_sections) if relevant_sections else wikitext

    links: set[str] = set()
    for match in re.finditer(r"\[\[([^|\]#]+)(?:#[^|\]]*)?(?:\|[^\]]*)?\]\]", text):
        title = match.group(1).replace("_", " ").strip()
        if not title or ":" in title:
            continue
        if _looks_like_non_article_link(title):
            continue
        links.add(title)
    return links


def _extract_relevant_sections(wikitext: str) -> list[str]:
    sections: list[str] = []
    current: list[str] | None = None
    current_level: int | None = None

    for line in wikitext.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        heading = _parse_wiki_heading(line)
        if heading:
            level, title = heading
            normalized_title = _normalize_section_title(title)
            if current is not None and current_level is not None and level <= current_level:
                sections.append("\n".join(current))
                current = None
                current_level = None
            if normalized_title in RELEVANT_LINK_SECTIONS:
                current = []
                current_level = level
            continue
        if current is not None:
            current.append(line)

    if current is not None:
        sections.append("\n".join(current))
    return sections


def _looks_like_non_article_link(title: str) -> bool:
    normalized = _normalize(title)
    if re.fullmatch(r"\d{3,4}(?: год(?: в литературе)?|(?:-е)? годы?)", normalized):
        return True
    if normalized in {
        "xx век",
        "футуризм",
        "символизм",
        "акмеизм",
        "имажинизм",
        "кубофутуризм",
        "эгофутуризм",
        "русский футуризм",
        "русский символизм",
        "русский авангард",
    }:
        return True
    return False


def resolve_titles(client: WikiClient, titles: Sequence[str]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for chunk in _chunked(list(dict.fromkeys(titles)), 50):
        data = client.get(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "info",
                "redirects": 1,
                "format": "json",
                "formatversion": "2",
            }
        )
        for page in data.get("query", {}).get("pages", []):
            if page.get("missing") or int(page.get("ns", -1)) != 0:
                continue
            resolved.append(
                {
                    "pageid": int(page["pageid"]),
                    "ns": int(page["ns"]),
                    "title": str(page["title"]),
                }
            )
    return resolved


def fetch_page(client: WikiClient, pageid: int) -> PageRecord | None:
    data = client.get(
        {
            "action": "query",
            "pageids": pageid,
            "prop": "extracts|info|categories",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "cllimit": "max",
            "inprop": "url",
            "redirects": 1,
            "format": "json",
            "formatversion": "2",
        }
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        return None

    return PageRecord(
        pageid=int(page["pageid"]),
        ns=int(page.get("ns", 0)),
        title=str(page["title"]),
        extract=str(page.get("extract", "")),
        fullurl=str(page.get("fullurl", "")),
        lastrevid=page.get("lastrevid"),
        categories=tuple(
            str(category["title"])
            for category in page.get("categories", [])
            if "title" in category
        ),
    )


def _unique_filename(title: str, pageid: int, used_filenames: set[str]) -> str:
    filename = safe_markdown_filename(title)
    if filename in used_filenames:
        stem = filename.removesuffix(".md")
        filename = f"{stem}-{pageid}.md"
    used_filenames.add(filename)
    return filename


def prepare_pages(
    client: WikiClient,
    candidates: dict[int, Candidate],
    *,
    generated_at: str,
    limit: int | None,
) -> tuple[list[IncludedPage], list[RejectedPage]]:
    included: list[IncludedPage] = []
    rejected: list[RejectedPage] = []
    used_filenames: set[str] = set()

    for candidate in sorted(candidates.values(), key=lambda item: item.title):
        page = fetch_page(client, candidate.pageid)
        sources = tuple(sorted(candidate.sources))
        if page is None:
            rejected.append(
                RejectedPage(
                    pageid=candidate.pageid,
                    title=candidate.title,
                    reason="missing_page",
                    sources=sources,
                )
            )
            continue

        decision = classify_page_for_inclusion(
            title=page.title,
            ns=page.ns,
            categories=page.categories,
            extract=page.extract,
            sources=sources,
        )
        if not decision.included:
            rejected.append(
                RejectedPage(
                    pageid=page.pageid,
                    title=page.title,
                    reason=decision.reason,
                    sources=sources,
                )
            )
            continue

        filename = _unique_filename(page.title, page.pageid, used_filenames)
        included.append(
            IncludedPage(
                page=page,
                filename=filename,
                reason=decision.reason,
                sources=sources,
                markdown=build_markdown(page, generated_at),
            )
        )
        if limit is not None and len(included) >= limit:
            break

    return included, rejected


def write_outputs(
    *,
    output_dir: Path,
    included: Sequence[IncludedPage],
    rejected: Sequence[RejectedPage],
    generated_at: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "source": "https://ru.wikipedia.org",
        "included_count": len(included),
        "rejected_count": len(rejected),
        "dry_run": dry_run,
        "included": [],
        "rejected": [],
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for item in included:
        target = output_dir / item.filename
        written = False
        status = "dry_run"
        if not dry_run:
            if target.exists() and not overwrite:
                status = "exists"
            else:
                target.write_text(item.markdown, encoding="utf-8")
                written = True
                status = "written"

        manifest["included"].append(
            {
                "pageid": item.page.pageid,
                "title": item.page.title,
                "url": item.page.fullurl,
                "revision": item.page.lastrevid,
                "file": str(target),
                "reason": item.reason,
                "sources": list(item.sources),
                "status": status,
                "written": written,
            }
        )

    for item in rejected:
        manifest["rejected"].append(
            {
                "pageid": item.pageid,
                "title": item.title,
                "reason": item.reason,
                "sources": list(item.sources),
            }
        )

    if not dry_run:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Russian Wikipedia biographies of Silver Age writers "
            "and save them as Markdown files."
        )
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output directory for Markdown files and manifest.json (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between MediaWiki API requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of accepted pages to write, useful for debugging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and classify pages without writing Markdown files or manifest.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be greater than zero")
    if args.delay < 0:
        raise ValueError("--delay must not be negative")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client = WikiClient(delay=args.delay)
    candidates: dict[int, Candidate] = {}

    collect_category_candidates(client, candidates)
    collect_search_candidates(client, candidates)
    collect_list_page_candidates(client, candidates)

    included, rejected = prepare_pages(
        client,
        candidates,
        generated_at=generated_at,
        limit=args.limit,
    )
    manifest = write_outputs(
        output_dir=Path(args.output),
        included=included,
        rejected=rejected,
        generated_at=generated_at,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print(f"Candidates: {len(candidates)}")
    print(f"Included: {manifest['included_count']}")
    print(f"Rejected: {manifest['rejected_count']}")
    if args.dry_run:
        print("Dry run: no files written.")
    else:
        print(f"Output: {Path(args.output)}")
        print(f"Manifest: {Path(args.output) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
