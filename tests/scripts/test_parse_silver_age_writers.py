from scripts.parse_silver_age_writers import (
    classify_page_for_inclusion,
    convert_wiki_headings_to_markdown,
    extract_relevant_links,
    safe_markdown_filename,
    strip_noise_tail_sections,
)


def test_safe_markdown_filename_keeps_cyrillic_and_removes_forbidden_chars():
    filename = safe_markdown_filename('Блок/Александр: <тест>?*')

    assert filename == "Блок Александр тест.md"


def test_convert_wiki_headings_to_markdown():
    text = "== Биография ==\nТекст\n=== Общие сведения ===\nЕщё текст"

    markdown = convert_wiki_headings_to_markdown(text)

    assert markdown == "## Биография\nТекст\n### Общие сведения\nЕщё текст"


def test_strip_noise_tail_sections_removes_tail_from_first_noise_heading():
    text = (
        "== Биография ==\n"
        "Основной текст\n"
        "== Примечания ==\n"
        "Шум\n"
        "== Литература ==\n"
        "Ещё шум"
    )

    stripped = strip_noise_tail_sections(text)

    assert stripped == "== Биография ==\nОсновной текст"


def test_classify_accepts_category_author():
    decision = classify_page_for_inclusion(
        title="Блок, Александр Александрович",
        ns=0,
        categories=(
            "Категория:Персоналии по алфавиту",
            "Категория:Писатели России XX века",
            "Категория:Русские поэты Серебряного века",
        ),
        extract=(
            "Александр Александрович Блок — русский писатель, "
            "одна из самых известных личностей Серебряного века."
        ),
        sources=(
            "category:Категория:Русские поэты Серебряного века",
            "seed_category:Категория:Русские поэты Серебряного века",
        ),
    )

    assert decision.included is True
    assert decision.reason == "seed_category_writer"


def test_classify_accepts_search_author_with_intro_signal():
    decision = classify_page_for_inclusion(
        title="Андреев, Леонид Николаевич",
        ns=0,
        categories=(
            "Категория:Персоналии по алфавиту",
            "Категория:Писатели России XX века",
        ),
        extract=(
            "Леонид Николаевич Андреев — русский писатель времён "
            "Серебряного века, представитель экспрессионизма."
        ),
        sources=('search:"русский писатель" "Серебряного века"',),
    )

    assert decision.included is True
    assert decision.reason == "search_silver_age_writer"


def test_classify_accepts_author_with_movement_signal():
    decision = classify_page_for_inclusion(
        title="Хлебников, Велимир",
        ns=0,
        categories=(
            "Категория:Персоналии по алфавиту",
            "Категория:Писатели России XX века",
            "Категория:Поэты русского авангарда",
            "Категория:Русский футуризм",
        ),
        extract=(
            "Велимир Хлебников — русский поэт и прозаик, "
            "один из основоположников русского футуризма."
        ),
        sources=("list_page:Русские поэты-футуристы",),
    )

    assert decision.included is True
    assert decision.reason == "list_page_movement_writer"


def test_classify_rejects_non_person_page():
    decision = classify_page_for_inclusion(
        title="Сатирикон (журнал)",
        ns=0,
        categories=(
            "Категория:Серебряный век",
            "Категория:Журналы Российской империи",
        ),
        extract="Сатирикон — русский еженедельный литературно-художественный журнал.",
        sources=("category:Категория:Серебряный век",),
    )

    assert decision.included is False
    assert decision.reason in {"non_person_title", "no_writer_signal", "no_person_signal"}


def test_classify_rejects_category_namespace():
    decision = classify_page_for_inclusion(
        title="Категория:Русские поэты-символисты",
        ns=14,
        categories=(),
        extract="",
        sources=(),
    )

    assert decision.included is False
    assert decision.reason == "unsupported_namespace"


def test_extract_relevant_links_uses_author_section():
    wikitext = (
        "[[Герцен, Александр Иванович]] outside relevant section\n"
        "== Список авторов ==\n"
        "* [[Маяковский, Владимир Владимирович|Владимир Маяковский]]\n"
        "* [[Хлебников, Велимир]]\n"
        "* [[1913 год в литературе]]\n"
        "* [[Категория:Русский футуризм]]\n"
        "== Примечания ==\n"
        "[[Футуризм]]\n"
    )

    links = extract_relevant_links(wikitext)

    assert "Маяковский, Владимир Владимирович" in links
    assert "Хлебников, Велимир" in links
    assert "Герцен, Александр Иванович" not in links
    assert "1913 год в литературе" not in links
    assert "Футуризм" not in links
