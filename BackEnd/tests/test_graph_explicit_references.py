"""REFERS_TO from an exact advisory-circular identifier in source text."""

from knowledge.graph_rag.hierarchy import build_structure

_QUOTE = "AC 00-24C"
_VERSIONED = "• 《Thunderstorms》（FAA AC 00-24C）"
_ICING_URL = "https://example.test/icing-notes"
_UNVERSIONED = "See AC 00-24 and " + _ICING_URL + " before departure."
_SELF = "This circular AC 00-24C restates its own number."


def _chunk(chunk_id: str, doc_id: str, content: str, page: int, title: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "document_key": doc_id,
        "document_title": title,
        "content": content,
        "original_format": "pdf",
        "page_number": page,
        "source_url": "https://example.test/" + doc_id,
    }


def _documents() -> dict:
    return {
        "weather-guide": {
            "documentKey": "weather-guide",
            "title": "Weather Guide",
            "name": "Weather Guide",
            "sourceUrl": "https://example.test/weather-guide",
            "storagePath": "library/weather-guide.pdf",
            "originalFormat": "pdf",
        },
        "ac-00-24c": {
            "documentKey": "ac-00-24c",
            "title": "AC00-24C: Thunderstorms",
            "name": "Thunderstorms",
            "sourceUrl": "https://example.test/ac-00-24c",
            "storagePath": "library/faa-ac00-24c.pdf",
            "originalFormat": "pdf",
        },
        "icing-notes": {
            "documentKey": "icing-notes",
            "title": "Icing Notes",
            "name": "Icing Notes",
            "sourceUrl": _ICING_URL,
            "storagePath": "library/icing-notes.pdf",
            "originalFormat": "pdf",
        },
    }


def _graph() -> dict:
    chunks = {
        "cite-versioned": _chunk("cite-versioned", "weather-guide", _VERSIONED, 4, "Weather Guide"),
        "cite-plain": _chunk("cite-plain", "weather-guide", _UNVERSIONED, 5, "Weather Guide"),
        "self-cite": _chunk("self-cite", "ac-00-24c", _SELF, 1, "AC00-24C: Thunderstorms"),
        "icing-body": _chunk("icing-body", "icing-notes", "No circular number is written here.", 1, "Icing Notes"),
    }
    return build_structure(_documents(), chunks)


def _document(graph: dict, label: str) -> dict:
    found = [
        node
        for node in graph["nodes"]
        if node.get("type") == "document" and node.get("label") == label
    ]
    assert len(found) == 1
    return found[0]


def test_explicit_ac_identifier_and_url_references() -> None:
    start = _VERSIONED.index(_QUOTE)
    assert start == 22
    assert _VERSIONED[start:start + len(_QUOTE)] == _QUOTE
    assert _QUOTE not in "AC00-24C: Thunderstorms"
    assert _QUOTE not in "library/faa-ac00-24c.pdf"
    assert _QUOTE not in _UNVERSIONED
    assert _QUOTE in _SELF

    graph = _graph()
    guide = _document(graph, "Weather Guide")
    circular = _document(graph, "AC00-24C: Thunderstorms")
    icing = _document(graph, "Icing Notes")
    refers = [edge for edge in graph["edges"] if edge.get("relationType") == "REFERS_TO"]
    assert len(refers) == 2
    by_source = {edge["source"]: edge for edge in refers}
    assert set(by_source) == {"chk:cite-versioned", "chk:cite-plain"}
    versioned = by_source["chk:cite-versioned"]
    plain = by_source["chk:cite-plain"]

    assert versioned["target"] == circular["id"]
    assert versioned["label"] == "REFERS_TO"
    assert versioned["edgeKind"] == "references"
    assert versioned["sourceIds"] == ["cite-versioned"]
    assert [item["method"] for item in versioned["provenance"]] == ["explicit_identifier"]
    item = versioned["provenance"][0]
    assert _VERSIONED[item["span"]["start"]:item["span"]["end"]] == item["quote"] == _QUOTE
    assert item["span"] == {"start": 22, "end": 31}
    assert item["sourceId"] == "cite-versioned"
    assert item["docId"] == guide["docId"]
    assert item["sourceUrl"] == "https://example.test/weather-guide"
    assert item["locator"]["page"] == 4

    assert plain["target"] == icing["id"]
    assert [item["method"] for item in plain["provenance"]] == ["explicit_url"]
    url_item = plain["provenance"][0]
    assert _UNVERSIONED[url_item["span"]["start"]:url_item["span"]["end"]] == url_item["quote"] == _ICING_URL
