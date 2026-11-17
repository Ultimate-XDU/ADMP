from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.evidence_scores import evidence_ranking_score
from utils.query_utils import compact_query_text, cross_lingual_query_terms, classify_query_intent
from utils.state_store import DEFAULT_STATE_FILE, load_state


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR.parent / "aerospace_docs"
STRUCTURAL_RECORD_DIR = BASE_DIR / "structural_records"
_CORPUS_DOCUMENTS_CACHE_LOCK = threading.RLock()


ARTICLE_LABEL_PATTERN = re.compile(
    r"(?:第\s*)?([0-9A-Za-z一二三四五六七八九十百千万两]+(?:\.[0-9A-Za-z一二三四五六七八九十百千万两]+)*)\s*条"
)
CODE_PATTERN = re.compile(
    r"""
    \b(?:
        (?:AC|CCAR|EASA|ATA|AMM|FIM|TSM|IPC|SRM)\s*[- ]?\s*\d{1,4}(?:[-. ]?\d{1,4}){0,3}[A-Z]?
        |FAA(?:\s*[- ]?\s*[A-Z])?\s*[- ]?\s*\d{3,5}(?:[-. ]?\d{1,4}){0,3}[A-Z]?
    )\b
    """,
    re.I | re.X,
)
CONCATENATED_CODE_PATTERN = re.compile(
    r"""
    \b(?:
        (?i:AC)\s*[- ]?\s*\d{1,4}(?:\.\d{1,4})?(?:[- ]?\d{1,4})?(?:[A-Z](?![a-z]))?
        |(?i:CCAR)\s*[- ]?\s*\d{1,4}(?:[-. ]?\d{1,4})?(?:[A-Z](?![a-z]))?
        |(?i:FAA)(?:\s*[- ]?\s*[A-Z])?\s*[- ]?\s*\d{3,5}(?:[-. ]?\d{1,4}){0,3}(?:[A-Z](?![a-z]))?
        |(?i:EASA|ATA|AMM|FIM|TSM|IPC|SRM)\s*[- ]?\s*\d{1,4}(?:[-. ]?\d{1,4}){0,3}(?:[A-Z](?![a-z]))?
    )
    """,
    re.X,
)
DOCUMENT_SUFFIX_PATTERN = re.compile(
    r"(?:管理)?(?:暂行)?(?:规定|规则|条例|办法|标准|指南|规范|手册|公约|通告|通知|程序|要求)$"
)

INTENT_MARKERS = (
    "可以作为",
    "能否作为",
    "是否可以作为",
    "应当在",
    "需要在",
    "包括哪些",
    "包含哪些",
    "有哪些",
    "如何",
    "怎么",
    "怎样",
    "是什么",
    "什么是",
    "多少",
    "区别",
    "不同",
    "比较",
    "对比",
    "规定",
    "要求",
    "需要",
    "应当",
    "必须",
    "包括",
    "包含",
    "分为",
    "适用于",
    "适用",
    "提交",
    "申请",
    "颁发",
    "签发",
    "办理",
    "处理",
    "确定",
    "检查",
    "检测",
    "报告",
)
QUESTION_WORDS = (
    "是什么",
    "有什么",
    "有哪些",
    "哪些",
    "哪几",
    "多少",
    "如何",
    "怎么",
    "怎样",
    "是否",
    "能否",
    "可以",
    "吗",
    "呢",
    "讲的",
    "规定了",
    "规定",
    "要求",
    "需要",
    "应当",
    "必须",
    "主要",
    "相关",
    "具体",
)
QUERY_STOPWORDS = {
    "中",
    "里",
    "内",
    "对",
    "和",
    "与",
    "及",
    "或",
    "的",
    "了",
    "吗",
    "呢",
    "什么",
    "内容",
    "建议",
    "目的",
    "要求",
    "规定",
    "相关",
    "主要",
    "哪些",
    "包括",
    "包含",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "by",
    "with",
    "and",
    "or",
    "as",
    "about",
}
GENERIC_TERMS = {
    "民用",
    "航空",
    "民航",
    "航空器",
    "飞机",
    "规章",
    "法规",
    "文件",
    "文档",
    "资料",
    "内容",
    "条款",
    "章节",
    "材料",
    "标准",
    "管理",
    "办法",
    "规定",
    "规则",
    "条例",
    "指南",
    "规范",
    "要求",
    "问题",
    "情况",
    "方面",
    "主要",
    "相关",
    "具体",
}


def compact_text(value: str) -> str:
    return compact_query_text(value)


def _query_tokens(value: str) -> Tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    tokens = re.findall(r"[a-z][a-z0-9]*(?:[-.][a-z0-9]+)*|\d+(?:[-.]\d+)*|[\u4e00-\u9fff]{2,}", normalized)
    cleaned: List[str] = []

    def push(token: str) -> None:
        compact = compact_text(token)
        if not compact or compact in QUERY_STOPWORDS or compact in GENERIC_TERMS:
            return
        if len(compact) < 2:
            return
        cleaned.append(compact[:28])

    for token in tokens:
        compact = compact_text(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{9,}", compact):
            parts = re.split(
                r"(?:的定义是什么|定义是什么|是什么|有什么|有哪些|哪些|哪几类|包括哪些|包含哪些|"
                r"应当在|需要在|多长时间|什么时间|什么时候|多久|如何|怎么|怎样|"
                r"可以作为|能否作为|是否可以|应当|需要|必须|告知|通知|报告|提交|申请|办理|处理|"
                r"规定|要求|建议|内容|相关|主要|具体|旅客|人员|单位|机构|材料|吗|呢)",
                compact,
            )
            for part in parts:
                push(part)
            continue
        push(token)
    return tuple(cleaned)


def _cross_lingual_intent_terms(value: str) -> Tuple[str, ...]:
    return tuple(compact_text(term) for term in cross_lingual_query_terms(value) if compact_text(term))


def _particleless_text(value: str) -> str:
    return re.sub(r"[的之其]", "", compact_text(value))


def _compact_forms(value: str) -> Tuple[str, ...]:
    base = compact_text(value)
    if not base:
        return ()
    forms = {base}
    particleless = _particleless_text(base)
    if particleless and particleless != base and len(particleless) >= 4:
        forms.add(particleless)
    if base.startswith("ac") and len(base) > 2:
        forms.add(base[2:])
    if base.startswith("ccar") and len(base) > 4:
        forms.add(base[4:])
    if base.startswith("faa") and len(base) > 3:
        forms.add(base[3:])
    if base.endswith("后") and len(base) >= 4:
        stem = base[:-1]
        forms.add(stem)
        forms.add(f"{stem}后")
        forms.add(f"{stem}后应当")
        forms.add(f"{stem}之日起")
        forms.add(f"自{stem}起")
    return tuple(sorted(forms, key=len, reverse=True))


def _term_matches(term: str, haystack: str) -> bool:
    haystack_forms = {compact_text(haystack)}
    particleless_haystack = _particleless_text(haystack)
    if particleless_haystack:
        haystack_forms.add(particleless_haystack)
    for form in _compact_forms(term):
        if form and any(form in haystack_form for haystack_form in haystack_forms):
            return True
        if len(form) >= 5 and re.fullmatch(r"[\u4e00-\u9fff]+", form):
            bigrams = {form[index : index + 2] for index in range(0, len(form) - 1)}
            if bigrams:
                hits = sum(1 for bigram in bigrams if any(bigram in haystack_form for haystack_form in haystack_forms))
                if hits / len(bigrams) >= 0.6:
                    return True
        if len(form) >= 8:
            grams = {form[index : index + 4] for index in range(0, len(form) - 3)}
            if grams:
                hits = sum(1 for gram in grams if any(gram in haystack_form for haystack_form in haystack_forms))
                threshold = 0.4 if re.fullmatch(r"[\u4e00-\u9fff]+", form) else 0.5
                if hits / len(grams) >= threshold:
                    return True
    return False


def _any_term_matches(terms: Sequence[str], haystack: str) -> bool:
    return any(term and _term_matches(term, haystack) for term in terms)


@dataclass(frozen=True)
class AliasMatchSignal:
    score: float = 0.0
    explicit: bool = False
    kind: str = ""


def _alias_exactly_named_in_query(alias: str, compact_query: str) -> bool:
    alias = compact_text(alias)
    return bool(alias and len(alias) >= 4 and alias in compact_query)


def _title_match_strength(expected: str, value: str) -> float:
    expected_codes = set(_extract_code_aliases(expected, include_concatenated=True))
    if expected_codes:
        value_codes = set(_extract_code_aliases(value, include_concatenated=True))
        return 1.0 if expected_codes & value_codes else 0.0

    expected_key = compact_text(expected)
    value_key = compact_text(value)
    if not expected_key or not value_key:
        return 0.0
    if expected_key == value_key:
        return 1.0
    shorter = min(len(expected_key), len(value_key))
    longer = max(len(expected_key), len(value_key))
    if shorter < 6:
        return 0.0
    if expected_key in value_key or value_key in expected_key:
        coverage = shorter / max(longer, 1)
        return coverage if coverage >= 0.75 else 0.0
    return 0.0


def document_matches_expected_title(doc: "CorpusDocument", expected: str) -> bool:
    values = (doc.doc_name, doc.title, *doc.aliases)
    return any(_title_match_strength(expected, value) > 0.0 for value in values)


def _alias_match_signal(alias: str, compact_query: str) -> AliasMatchSignal:
    alias = compact_text(alias)
    if not alias:
        return AliasMatchSignal()
    if _alias_exactly_named_in_query(alias, compact_query):
        return AliasMatchSignal(220.0 + min(len(alias), 80), True, "exact_title")
    if len(compact_query) >= 6 and compact_query in alias:
        coverage = len(compact_query) / max(len(alias), 1)
        if coverage >= 0.72:
            return AliasMatchSignal(160.0 + min(len(compact_query), 60), True, "query_in_title")
    if _term_matches(alias, compact_query):
        return AliasMatchSignal(min(28.0, 10.0 + len(alias) * 0.45), False, "fuzzy_title")
    return AliasMatchSignal()


def _quoted_alias_match_signal(fragment: str, alias: str, *, shared_alias: bool) -> AliasMatchSignal:
    fragment = compact_text(fragment)
    alias = compact_text(alias)
    if not fragment or not alias:
        return AliasMatchSignal()
    if fragment == alias:
        if shared_alias:
            return AliasMatchSignal(90.0 + min(len(alias), 40), False, "quoted_shared_alias")
        return AliasMatchSignal(1000.0 + min(len(alias), 80), True, "quoted_title")
    if fragment in alias or alias in fragment:
        shorter = min(len(fragment), len(alias))
        longer = max(len(fragment), len(alias))
        coverage = shorter / max(longer, 1)
        if coverage >= 0.9 and not shared_alias:
            return AliasMatchSignal(1000.0 + shorter, True, "quoted_title")
        if coverage >= 0.75:
            return AliasMatchSignal(120.0 + min(shorter, 40), False, "quoted_partial_title")
    return AliasMatchSignal()


def _definition_focus_matches(term: str, row: Dict, haystack: str) -> bool:
    subject = compact_text(term)
    if not subject:
        return False

    role = str(row.get("chunk_role") or row.get("semantic_role") or "").lower()
    if subject in {compact_text("定义"), compact_text("含义")}:
        return "definition" in role or "定义" in haystack or "是指" in haystack or "指" in haystack

    raw_text = unicodedata.normalize(
        "NFKC",
        "\n".join(
            str(row.get(key) or "")
            for key in ("heading", "section_title", "highlight_text", "title", "label", "content", "body_text")
        ),
    )
    compact_raw = compact_text(raw_text)
    if not compact_raw:
        return False

    subject_pattern = re.escape(term)
    if re.search(rf"(^|[\s\(（;；。]){subject_pattern}\s*(?:[:：]|是指|指|是|为)", raw_text):
        return True

    compact_markers = (
        f"{subject}是指",
        f"{subject}指",
        f"{subject}是",
        f"{subject}为",
    )
    if any(marker in compact_raw for marker in compact_markers):
        return True

    for key in ("heading", "section_title", "highlight_text", "title", "label"):
        if compact_text(str(row.get(key) or "")) == subject:
            return True

    return False


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _record_path_for_doc(doc_name: str) -> Path:
    return STRUCTURAL_RECORD_DIR / f"{Path(doc_name).stem}.json"


def _title_aliases(title: str, storage_file_name: str) -> Tuple[str, ...]:
    stem = Path(storage_file_name or title).stem
    values = {title, stem, storage_file_name}
    pending = list(values)

    def add_value(candidate: str) -> None:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in values:
            values.add(candidate)
            pending.append(candidate)

    while pending:
        value = pending.pop(0)
        for inner in re.findall(r"《([^》]{3,100})》", value):
            add_value(inner)
        for inner in re.findall(r"[（(]([^）)]{3,100})[）)]", value):
            if not re.fullmatch(r"[0-9年月日\s.-]+", inner):
                add_value(inner)
        without_brackets = re.sub(r"[（(][^）)]{2,80}[）)]", "", value)
        if without_brackets != value:
            add_value(without_brackets)
        stripped = DOCUMENT_SUFFIX_PATTERN.sub("", value)
        if stripped != value:
            add_value(stripped)
        modified = re.sub(r"^(?:[^《]{0,30}关于(?:修订|修改)《)([^》]{3,100})(?:》.*)$", r"\1", value)
        if modified != value:
            add_value(modified)

    aliases = {compact_text(value) for value in values}
    return tuple(sorted((alias for alias in aliases if len(alias) >= 4), key=len, reverse=True))


def _code_aliases_from_title(title: str, storage_file_name: str) -> Tuple[str, ...]:
    aliases = set()
    values = {
        str(title or ""),
        str(storage_file_name or ""),
        Path(str(title or "")).stem,
        Path(str(storage_file_name or "")).stem,
    }
    for value in values:
        aliases.update(_extract_code_aliases(value, include_concatenated=True))
    return tuple(sorted(aliases, key=len, reverse=True))


def code_query_variants(code: str) -> Tuple[str, ...]:
    compact = compact_text(code)
    if not compact:
        return ()
    variants = {compact}
    match = re.match(r"^([a-z]+)([0-9].*)$", compact)
    if match:
        prefix, suffix = match.groups()
        if prefix == "ac":
            compact_suffix = suffix.replace(".", "")
            if "." in suffix:
                family, rest = suffix.split(".", 1)
                variants.add(f"ac{family}.{rest}")
                if len(rest) > 2 and not rest[2:].startswith("0"):
                    variants.add(f"ac{family}.{rest[:2]}-{rest[2:]}")
                elif rest:
                    variants.add(f"ac{family}-{rest}")
            elif re.match(r"^\d{4,}[a-z]?$", compact_suffix):
                letter = compact_suffix[-1] if compact_suffix[-1].isalpha() else ""
                digits = compact_suffix[:-1] if letter else compact_suffix
                if letter and len(digits) >= 4:
                    tail_start = 3
                else:
                    tail_start = 3 if len(digits) >= 5 and not digits[3:].startswith("0") else 2
                head = digits[:tail_start]
                tail = digits[tail_start:] + letter
                if tail:
                    variants.add(f"ac{head}-{tail}")
                    variants.add(f"ac {head}-{tail}")
            else:
                variants.add(f"ac-{suffix}")
                variants.add(f"ac {suffix}")
        elif prefix == "ccar":
            variants.add(f"ccar-{suffix}")
            variants.add(f"ccar {suffix}")
        elif prefix.startswith("faa"):
            faa_suffix = suffix
            designator = prefix[3:]
            if designator:
                variants.add(f"faa-{designator}-{faa_suffix}")
                variants.add(f"faa {designator} {faa_suffix}")
            else:
                variants.add(f"faa-{faa_suffix}")
                variants.add(f"faa {faa_suffix}")
            if len(faa_suffix) > 4:
                head, tail = faa_suffix[:4], faa_suffix[4:]
                if designator:
                    variants.add(f"faa-{designator}-{head}-{tail}")
                else:
                    variants.add(f"faa-{head}-{tail}")
        else:
            variants.add(f"{prefix}-{suffix}")
            variants.add(f"{prefix} {suffix}")
        variants.add(suffix)
    return tuple(sorted(variants, key=len, reverse=True))


def _extract_code_aliases(text: str, *, include_concatenated: bool = False) -> Tuple[str, ...]:
    aliases = set()
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    patterns = [CODE_PATTERN]
    if include_concatenated:
        patterns.append(CONCATENATED_CODE_PATTERN)
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            alias = compact_text(match.group(0))
            if len(alias) >= 4:
                aliases.add(alias)
    return tuple(sorted(aliases, key=len, reverse=True))


def _query_codes(query: str) -> Tuple[str, ...]:
    return _extract_code_aliases(query)


def _is_modifier_title(title: str) -> bool:
    return any(token in str(title or "") for token in ("关于修改", "关于修订", "修改《", "修订《"))


def _doc_base_aliases(doc: "CorpusDocument") -> Tuple[str, ...]:
    aliases = [alias for alias in doc.aliases if alias]
    if not _is_modifier_title(doc.title):
        return tuple(aliases)

    title_key = compact_text(doc.title)
    storage_key = compact_text(doc.storage_file_name)
    return tuple(alias for alias in aliases if alias and alias not in {title_key, storage_key})


def _prefer_canonical_document(doc: "CorpusDocument", docs: Sequence["CorpusDocument"], alias: str) -> Optional["CorpusDocument"]:
    if not alias or not _is_modifier_title(doc.title):
        return None
    candidates = [
        candidate
        for candidate in docs
        if candidate.doc_name != doc.doc_name and not _is_modifier_title(candidate.title) and alias in candidate.aliases
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.is_synced, item.chunk_count, len(item.title)), reverse=True)
    return candidates[0]


def _ccar_part_from_code(code: str) -> str:
    match = re.match(r"^ccar[- ]?(\d{2,4})(?:r\d+)?$", compact_text(code), re.I)
    return match.group(1) if match else ""


def _doc_matches_ccar_part(doc: "CorpusDocument", part: str) -> bool:
    if not part:
        return False
    values = [doc.title, doc.storage_file_name, *doc.aliases, *doc.code_aliases]
    for value in values:
        compact = compact_text(value)
        if not compact:
            continue
        if re.search(rf"(?:^|[^0-9]){re.escape(part)}(?:\.[0-9]+|r\d+|[^0-9]|$)", compact):
            return True
    record_path = STRUCTURAL_RECORD_DIR / f"{Path(doc.storage_file_name).stem}.json"
    if record_path.exists():
        try:
            sample = record_path.read_text(encoding="utf-8")[:5000]
        except Exception:
            sample = ""
        if re.search(rf"(?:CCAR[-\s]?{re.escape(part)}|第\s*{re.escape(part)}\.\d+\s*条|{re.escape(part)}\.\d+)", sample, flags=re.I):
            return True
    return False


def _canonical_doc_for_ccar_part(part: str, docs: Sequence["CorpusDocument"]) -> Optional["CorpusDocument"]:
    if not part:
        return None
    candidates = [doc for doc in docs if _doc_matches_ccar_part(doc, part)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda doc: (
            doc.is_synced,
            not _is_modifier_title(doc.title),
            doc.chunk_count,
            -len(doc.title),
        ),
        reverse=True,
    )
    return candidates[0]


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    storage_file_name: str
    status: str
    chunk_count: int = 0
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    code_aliases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def doc_name(self) -> str:
        return self.storage_file_name

    @property
    def is_synced(self) -> bool:
        return self.status.upper() == "SYNCED" and self.chunk_count != 0


@dataclass(frozen=True)
class DocumentCandidateMatch:
    score: float
    doc: CorpusDocument
    alias: str
    explicit: bool
    match_kind: str


@dataclass(frozen=True)
class RetrievalTarget:
    doc_name: str
    title: str
    status: str
    chunk_count: int
    confidence: float
    explicit: bool
    matched_alias: str = ""
    match_kind: str = ""

    @property
    def is_synced(self) -> bool:
        return self.status.upper() == "SYNCED" and self.chunk_count != 0


@dataclass(frozen=True)
class RetrievalContract:
    query: str
    intent: str
    targets: Tuple[RetrievalTarget, ...] = field(default_factory=tuple)
    query_codes: Tuple[str, ...] = field(default_factory=tuple)
    article_id: str = ""
    article_label: str = ""
    article_variants: Tuple[str, ...] = field(default_factory=tuple)
    required_terms: Tuple[str, ...] = field(default_factory=tuple)
    focus_terms: Tuple[str, ...] = field(default_factory=tuple)
    hard_target: bool = False
    allow_multi_doc: bool = False
    requires_table: bool = False
    requires_strict_coverage: bool = False
    requires_all_focus_terms: bool = False
    block_reason: str = ""

    @property
    def target_doc_names(self) -> Tuple[str, ...]:
        return tuple(target.doc_name for target in self.targets)

    @property
    def synced_target_doc_names(self) -> Tuple[str, ...]:
        return tuple(target.doc_name for target in self.targets if target.is_synced)

    @property
    def has_hard_constraints(self) -> bool:
        return bool(self.block_reason or self.hard_target or self.article_variants or self.requires_strict_coverage)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["target_doc_names"] = list(self.target_doc_names)
        payload["synced_target_doc_names"] = list(self.synced_target_doc_names)
        payload["hard_target_doc_names"] = list(hard_target_doc_names(self))
        payload["specific_focus_target_doc_names"] = list(specific_focus_target_doc_names(self))
        return payload


def _focus_term_core(value: str) -> str:
    compact = compact_text(value)
    return re.sub(r"^(?:的|对|关于|针对|有关|中的|里|里的)", "", compact)


def specific_focus_target_doc_names(contract: RetrievalContract) -> Tuple[str, ...]:
    """Return supplemental focus-target docs that are specific enough to keep."""
    focus_cores = [core for core in (_focus_term_core(term) for term in contract.focus_terms) if len(core) >= 4]
    if not focus_cores:
        return ()

    names: List[str] = []
    seen: set[str] = set()
    for target in contract.targets:
        if target.explicit or target.match_kind != "focus_term" or not target.is_synced:
            continue
        target_text = compact_text(f"{target.title} {target.doc_name}")
        target_match_text = re.sub(r"[的之]", "", target_text)
        matched = any(
            core and (core in target_text or re.sub(r"[的之]", "", core) in target_match_text)
            for core in focus_cores
        )
        if matched and target.doc_name not in seen:
            names.append(target.doc_name)
            seen.add(target.doc_name)
    return tuple(names)


def contract_needs_multi_target_recall(contract: RetrievalContract) -> bool:
    if len(contract.synced_target_doc_names) <= 1:
        return False
    if contract.allow_multi_doc:
        return True
    return bool(contract.hard_target and specific_focus_target_doc_names(contract))


def hard_target_doc_names(contract: RetrievalContract) -> Tuple[str, ...]:
    """Return the document allow-list used to validate hard-target evidence."""
    if not contract.hard_target:
        return contract.synced_target_doc_names

    names: List[str] = []
    seen: set[str] = set()
    for target in contract.targets:
        if target.explicit and target.is_synced and target.doc_name not in seen:
            names.append(target.doc_name)
            seen.add(target.doc_name)
    if contract_needs_multi_target_recall(contract):
        for doc_name in specific_focus_target_doc_names(contract):
            if doc_name not in seen:
                names.append(doc_name)
                seen.add(doc_name)
    return tuple(names) or contract.synced_target_doc_names


@dataclass(frozen=True)
class EvidenceValidation:
    passed: bool
    score: float
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    missing_terms: Tuple[str, ...] = field(default_factory=tuple)
    matched_terms: Tuple[str, ...] = field(default_factory=tuple)


def _coverage_threshold(term_count: int, *, default: float = 0.45) -> float:
    if term_count <= 0:
        return 0.0
    if term_count <= 2:
        return 1.0
    if term_count <= 5:
        return 0.6
    return default


@dataclass(frozen=True)
class EvidenceCoveragePolicy:
    required_threshold: float
    focus_threshold: float
    strict_required: bool
    strict_focus: bool
    all_focus_required: bool

    @classmethod
    def from_contract(cls, contract: RetrievalContract) -> "EvidenceCoveragePolicy":
        required_count = len(contract.required_terms)
        focus_count = len(contract.focus_terms)
        required_threshold = _coverage_threshold(required_count)
        focus_threshold = _coverage_threshold(focus_count, default=0.5)
        if contract.hard_target:
            required_threshold = min(required_threshold, 0.6)
            focus_threshold = min(focus_threshold, 0.5)
        return cls(
            required_threshold=required_threshold,
            focus_threshold=focus_threshold,
            strict_required=contract.requires_strict_coverage,
            strict_focus=contract.requires_strict_coverage,
            all_focus_required=contract.requires_all_focus_terms,
        )


_corpus_documents_cache: Tuple[CorpusDocument, ...] = ()
_corpus_documents_cache_stamp: Optional[Tuple[int, ...]] = None


def _load_corpus_documents_cached() -> Tuple[CorpusDocument, ...]:
    records: Dict[str, CorpusDocument] = {}
    payload = load_state()
    for item in payload.get("documents", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        storage = str(item.get("storageFileName") or item.get("fileName") or "").strip()
        if not storage:
            continue
        title = str(item.get("fileName") or Path(storage).stem).strip()
        records[storage] = CorpusDocument(
            title=title,
            storage_file_name=storage,
            status=str(item.get("status") or "").strip().upper() or "UNKNOWN",
            chunk_count=_safe_int(item.get("chunkCount"), 0),
            aliases=_title_aliases(title, storage),
            code_aliases=_code_aliases_from_title(title, storage),
        )

    if DOC_DIR.exists():
        for path in DOC_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx"} or path.name in records:
                continue
            status = "SYNCED" if _record_path_for_doc(path.name).exists() else "UNKNOWN"
            records[path.name] = CorpusDocument(
                title=path.stem,
                storage_file_name=path.name,
                status=status,
                chunk_count=1 if status == "SYNCED" else 0,
                aliases=_title_aliases(path.stem, path.name),
                code_aliases=_code_aliases_from_title(path.stem, path.name),
            )

    return tuple(sorted(records.values(), key=lambda item: item.storage_file_name))


def _corpus_cache_stamp() -> Tuple[int, ...]:
    stamps: List[int] = []
    for target in (DEFAULT_STATE_FILE, DOC_DIR):
        try:
            stat = target.stat()
        except OSError:
            stamps.append(0)
        else:
            stamps.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


def load_corpus_documents() -> Tuple[CorpusDocument, ...]:
    global _corpus_documents_cache, _corpus_documents_cache_stamp
    with _CORPUS_DOCUMENTS_CACHE_LOCK:
        stamp = _corpus_cache_stamp()
        if _corpus_documents_cache_stamp != stamp:
            _corpus_documents_cache = _load_corpus_documents_cached()
            _corpus_documents_cache_stamp = stamp
        return _corpus_documents_cache


def clear_contract_caches() -> None:
    global _corpus_documents_cache, _corpus_documents_cache_stamp
    with _CORPUS_DOCUMENTS_CACHE_LOCK:
        _corpus_documents_cache = ()
        _corpus_documents_cache_stamp = None


def _quoted_fragments(query: str) -> Tuple[str, ...]:
    fragments = []
    for pattern in (r"《([^》]{2,120})》", r"[\"']([^\"']{2,120})[\"']"):
        fragments.extend(match.group(1) for match in re.finditer(pattern, query))
    return tuple(fragments)


def _looks_like_comparison(query: str) -> bool:
    compact = compact_text(query)
    return any(token in compact for token in ("区别", "不同", "差异", "比较", "对比", "侧重点", "异同"))


def _prefer_quoted_comparison_targets(query: str, targets: Sequence[RetrievalTarget]) -> List[RetrievalTarget]:
    quoted = [compact_text(fragment) for fragment in _quoted_fragments(query)]
    quoted = [fragment for fragment in quoted if len(fragment) >= 4]
    if len(quoted) < 2:
        return list(targets)

    selected: List[RetrievalTarget] = []
    used_doc_names: set[str] = set()
    for fragment in quoted:
        candidates = [
            target
            for target in targets
            if target.doc_name not in used_doc_names
            and (
                fragment == compact_text(target.matched_alias)
                or fragment == compact_text(target.title)
                or fragment == compact_text(Path(target.doc_name).stem)
                or fragment in compact_text(target.title)
            )
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda target: (
                compact_text(target.title) == fragment or compact_text(Path(target.doc_name).stem) == fragment,
                not _is_modifier_title(target.title),
                target.explicit,
                target.is_synced,
                target.chunk_count,
                target.confidence,
            ),
            reverse=True,
        )
        selected.append(candidates[0])
        used_doc_names.add(candidates[0].doc_name)

    if len(selected) < 2:
        return list(targets)
    return selected


def _intent_from_query(query: str, article_label: str) -> str:
    return classify_query_intent(query, article_label=article_label).intent


def _chinese_number_to_int(value: str) -> Optional[int]:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if not value or not all(char in digits or char in units for char in value):
        return None
    total = 0
    current = 0
    for char in value:
        if char in digits:
            current = digits[char]
        else:
            unit = units[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
    return total + current


def _article_variants(article_id: str, article_label: str) -> Tuple[str, ...]:
    variants = {compact_text(article_id), compact_text(article_label)}
    if article_id:
        variants.add(compact_text(f"第{article_id}条"))
        variants.add(compact_text(f"{article_id}条"))
        as_int = _chinese_number_to_int(article_id)
        if as_int is not None:
            variants.add(str(as_int))
            variants.add(compact_text(f"第{as_int}条"))
            variants.add(compact_text(f"{as_int}条"))
    return tuple(sorted((value for value in variants if value), key=len, reverse=True))


def _extract_article(query: str) -> Tuple[str, str, Tuple[str, ...]]:
    match = ARTICLE_LABEL_PATTERN.search(query)
    if not match:
        return "", "", ()
    article_id = re.sub(r"\s+", "", match.group(1))
    article_label = f"第{article_id}条"
    return article_id, article_label, _article_variants(article_id, article_label)


def _document_match_candidates(query: str, docs: Sequence[CorpusDocument], query_codes: Sequence[str]) -> List[RetrievalTarget]:
    compact_query = compact_text(query)
    quoted = tuple(compact_text(fragment) for fragment in _quoted_fragments(query))
    query_code_set = set(query_codes)
    matches: Dict[str, DocumentCandidateMatch] = {}
    alias_counts: Dict[str, int] = {}
    for doc in docs:
        for alias in doc.aliases:
            alias_key = compact_text(alias)
            if alias_key:
                alias_counts[alias_key] = alias_counts.get(alias_key, 0) + 1

    for code in query_codes:
        part = _ccar_part_from_code(code)
        if not part:
            continue
        ccar_doc = _canonical_doc_for_ccar_part(part, docs)
        if ccar_doc is None:
            continue
        score = 34.0 + min(len(part), 4)
        current = matches.get(ccar_doc.doc_name)
        if current is None or score > current.score:
            matches[ccar_doc.doc_name] = DocumentCandidateMatch(score, ccar_doc, f"ccar-{part}", True, "code")

    for doc in docs:
        for alias in doc.aliases:
            matched_doc = doc
            explicit = False
            score = 0.0
            match_kind = ""
            query_quoted_match = False
            if quoted:
                for fragment in quoted:
                    quoted_signal = _quoted_alias_match_signal(
                        fragment,
                        alias,
                        shared_alias=alias_counts.get(compact_text(alias), 0) > 1,
                    )
                    if quoted_signal.score > score:
                        query_quoted_match = quoted_signal.kind == "quoted_title"
                        explicit = quoted_signal.explicit
                        score = quoted_signal.score
                        match_kind = quoted_signal.kind
            alias_signal = _alias_match_signal(alias, compact_query)
            if alias_signal.score > score:
                score = alias_signal.score
                explicit = alias_signal.explicit
                match_kind = alias_signal.kind
            if score <= 0.0:
                continue
            canonical_doc = _prefer_canonical_document(doc, docs, alias)
            if canonical_doc and query_quoted_match:
                matched_doc = canonical_doc
                score += 1.0
            elif match_kind == "fuzzy_title" and _is_modifier_title(doc.title) and alias in _doc_base_aliases(doc):
                score -= 12.0
            if matched_doc.status == "SYNCED":
                score += 0.25
            current = matches.get(matched_doc.doc_name)
            if current is None or score > current.score:
                matches[matched_doc.doc_name] = DocumentCandidateMatch(score, matched_doc, alias, explicit, match_kind)

        for code_alias in doc.code_aliases:
            if code_alias and code_alias in query_code_set:
                score = 24.0 + len(code_alias) * 0.2
                current = matches.get(doc.doc_name)
                if current is None or score > current.score:
                    matches[doc.doc_name] = DocumentCandidateMatch(score, doc, code_alias, True, "code")

    ranked = sorted(matches.values(), key=lambda item: item.score, reverse=True)
    if not ranked:
        return []

    comparison = _looks_like_comparison(query)
    top_score = ranked[0].score
    targets: List[RetrievalTarget] = []
    for match in ranked:
        score = match.score
        keep = score >= max(6.0, top_score * (0.45 if comparison else 0.82))
        if match.explicit:
            keep = True
        if not keep:
            continue
        targets.append(
            RetrievalTarget(
                doc_name=match.doc.doc_name,
                title=match.doc.title,
                status=match.doc.status,
                chunk_count=match.doc.chunk_count,
                confidence=round(min(1.0, max(0.0, score / 40.0)), 4),
                explicit=match.explicit,
                matched_alias=match.alias,
                match_kind=match.match_kind,
            )
        )
        if not comparison and match.explicit:
            break
        if not comparison and len(targets) >= 3:
            break
        if len(targets) >= 4:
            break
    if comparison:
        compact_query_terms = [term for term in re.split(r"和|与|及|、|,|，|以及", query) if term.strip()]
        existing_doc_names = {target.doc_name for target in targets}
        for term in compact_query_terms:
            term_compact = compact_text(_strip_query_piece(term))
            if len(term_compact) < 4:
                continue
            term_matches = [
                (len(alias), doc, alias)
                for doc in docs
                if doc.doc_name not in existing_doc_names
                for alias in _doc_base_aliases(doc)
                if alias and (term_compact in alias or alias in term_compact)
            ]
            if not term_matches:
                continue
            term_matches.sort(key=lambda item: (item[0], item[1].is_synced, item[1].chunk_count), reverse=True)
            _, doc, alias = term_matches[0]
            targets.append(
                RetrievalTarget(
                    doc_name=doc.doc_name,
                    title=doc.title,
                    status=doc.status,
                    chunk_count=doc.chunk_count,
                    confidence=round(min(1.0, max(0.0, len(alias) / 40.0)), 4),
                    explicit=False,
                    matched_alias=alias,
                    match_kind="comparison_term",
                )
            )
            existing_doc_names.add(doc.doc_name)
            if len(targets) >= 4:
                break
    if comparison:
        targets = _prefer_quoted_comparison_targets(query, targets)
    return targets


def _supplement_targets_from_terms(
    docs: Sequence[CorpusDocument],
    existing_targets: Sequence[RetrievalTarget],
    terms: Sequence[str],
    *,
    max_targets: int = 4,
) -> Tuple[RetrievalTarget, ...]:
    targets = list(existing_targets)
    existing_doc_names = {target.doc_name for target in targets}
    for term in terms:
        term_compact = compact_text(term)
        if len(term_compact) < 4:
            continue
        matches: List[Tuple[int, CorpusDocument, str]] = []
        for doc in docs:
            if doc.doc_name in existing_doc_names:
                continue
            for alias in doc.aliases:
                if len(alias) < 4:
                    continue
                if _term_matches(term_compact, alias) or _term_matches(alias, term_compact):
                    matches.append((len(alias), doc, alias))
        if not matches:
            continue
        matches.sort(key=lambda item: (item[1].is_synced, item[0], item[1].chunk_count), reverse=True)
        _, doc, alias = matches[0]
        targets.append(
            RetrievalTarget(
                doc_name=doc.doc_name,
                title=doc.title,
                status=doc.status,
                chunk_count=doc.chunk_count,
                confidence=round(min(1.0, max(0.0, len(alias) / 40.0)), 4),
                explicit=False,
                matched_alias=alias,
                match_kind="focus_term",
            )
        )
        existing_doc_names.add(doc.doc_name)
        if len(targets) >= max_targets:
            break
    return tuple(targets)


def _strip_targets_and_articles(query: str, targets: Sequence[RetrievalTarget], article_variants: Sequence[str]) -> str:
    compact = compact_text(query)
    for target in targets:
        for alias in _title_aliases(target.title, target.doc_name):
            compact = compact.replace(alias, "")
    for variant in article_variants:
        compact = compact.replace(compact_text(variant), "")
    return compact


def _strip_query_piece(value: str) -> str:
    value = compact_text(value)
    if re.search(r"[A-Za-z]", value):
        value = re.sub(r"[\u4e00-\u9fff]+", "", value)
    for prefix in ("对于", "关于", "针对", "依据", "根据", "对", "在", "从"):
        prefix_key = compact_text(prefix)
        if value.startswith(prefix_key) and len(value) > len(prefix_key) + 1:
            value = value[len(prefix_key) :]
            break
    for marker in QUESTION_WORDS:
        marker_key = compact_text(marker)
        if marker_key:
            value = value.replace(marker_key, "")
    for suffix in (
        "是什么",
        "有什么",
        "有哪些",
        "哪些",
        "哪几类",
        "包括哪些",
        "包含哪些",
        "什么建议",
        "内容",
        "建议",
        "要求",
        "规定",
        "目的",
        "中的",
        "中",
        "里的",
        "里",
        "方面",
        "相关",
        "具体",
    ):
        suffix_key = compact_text(suffix)
        if value.endswith(suffix_key) and len(value) > len(suffix_key) + 1:
            value = value[: -len(suffix_key)]
            break
    return value


def _extract_query_terms(query: str, targets: Sequence[RetrievalTarget], article_variants: Sequence[str], query_codes: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    readable_working = _strip_targets_and_articles(query, targets, article_variants)
    token_candidates = list(_query_tokens(readable_working))
    intent_candidates = list(_cross_lingual_intent_terms(readable_working))
    working = compact_text(readable_working)
    required = list(query_codes)
    focus_candidates: List[str] = []
    for code in query_codes:
        for variant in code_query_variants(code):
            working = working.replace(compact_text(variant), "|")
    for marker in INTENT_MARKERS:
        working = working.replace(compact_text(marker), "|")
    if not targets:
        working = re.sub(r"(应当|需要|必须)?(在)?(多长时间|多久|什么时候|什么时间|哪些|什么|如何|怎么|怎样)(内)?", "|", working)
        working = re.sub(r"(告知|通知|报告|提交|申请|办理|处理|规定|要求|建议|内容|旅客|人员|单位|机构)", "|", working)
    working = re.sub(r"[的了吗呢中里上于和与及或、：:；;，,。?？（）()【】\[\]]+", "|", working)

    for piece in working.split("|"):
        value = _strip_query_piece(piece)
        for marker in QUESTION_WORDS:
            value = value.replace(compact_text(marker), "")
        value = _strip_query_piece(value)
        if len(value) < 2 or value in GENERIC_TERMS:
            continue
        focus_candidates.append(value[:28])
    focus_candidates.extend(token_candidates)
    focus_candidates.extend(intent_candidates)

    normalized_required: List[str] = []
    for value in sorted(required, key=len, reverse=True):
        value = compact_text(value)
        if not value or value in GENERIC_TERMS:
            continue
        if any(value != existing and value in existing for existing in normalized_required):
            continue
        normalized_required.append(value)

    normalized_focus: List[str] = []
    for value in sorted(focus_candidates, key=len, reverse=True):
        value = compact_text(value)
        if not value or value in GENERIC_TERMS:
            continue
        if value in normalized_required:
            continue
        if value in QUERY_STOPWORDS:
            continue
        if any(value != existing and value in existing for existing in normalized_focus):
            continue
        normalized_focus = [existing for existing in normalized_focus if not (existing != value and existing in value)]
        normalized_focus.append(value)
        if len(normalized_focus) >= 8:
            break
    return tuple(normalized_required), tuple(normalized_focus)


def _relax_focus_terms(intent: str, focus_terms: Sequence[str]) -> Tuple[str, ...]:
    if intent == "count":
        return ()
    if intent == "definition_or_explanation":
        return tuple(term for term in focus_terms if term not in {compact_text("定义"), compact_text("含义")})
    return tuple(focus_terms)


def _strict_coverage_required(
    query: str,
    intent: str,
    targets: Sequence[RetrievalTarget],
    article_variants: Sequence[str] = (),
) -> bool:
    # A verified article identifier is already an exact source constraint.  The
    # remaining natural-language wording describes what to explain from that
    # article and must not reject the article merely because interrogative words
    # such as "规定了什么" do not appear verbatim in its text.
    if article_variants:
        return False
    compact = compact_text(query)
    if any(token in compact for token in ("是否", "能否", "可以作为", "能不能", "可不可以")):
        return True
    if intent == "table":
        return True
    return not targets


def _all_focus_required(query: str, intent: str) -> bool:
    compact = compact_text(query)
    return bool(
        intent == "table"
        or any(token in compact for token in ("是否", "能否", "可以作为", "能不能", "可不可以"))
    )


def build_retrieval_contract(query: str) -> RetrievalContract:
    docs = load_corpus_documents()
    compact_query = compact_text(query)
    article_id, article_label, article_variants = _extract_article(query)
    query_codes = _query_codes(query)
    intent = _intent_from_query(query, article_label)
    targets = tuple(_document_match_candidates(query, docs, query_codes))
    allow_multi_doc = intent == "comparison"
    hard_target = bool(targets and targets[0].explicit and not allow_multi_doc)
    required_terms, focus_terms = _extract_query_terms(query, targets, article_variants, query_codes)
    focus_terms = _relax_focus_terms(intent, focus_terms)
    targets = _supplement_targets_from_terms(docs, targets, focus_terms)
    hard_target = bool(targets and targets[0].explicit and not allow_multi_doc)

    block_reason = ""
    explicitly_named_targets = tuple(
        target
        for target in targets
        if target.explicit and target.match_kind in {"exact_title", "quoted_title", "query_in_title"}
    )
    if explicitly_named_targets and not any(target.is_synced for target in explicitly_named_targets):
        block_reason = "target_document_not_indexed"
    elif hard_target and not any(target.is_synced for target in targets):
        block_reason = "target_document_not_indexed"

    return RetrievalContract(
        query=query,
        intent=intent,
        targets=targets,
        query_codes=query_codes,
        article_id=article_id,
        article_label=article_label,
        article_variants=article_variants,
        required_terms=required_terms,
        focus_terms=focus_terms,
        hard_target=hard_target,
        allow_multi_doc=allow_multi_doc,
        requires_table=intent == "table",
        requires_strict_coverage=_strict_coverage_required(query, intent, targets, article_variants),
        requires_all_focus_terms=_all_focus_required(query, intent),
        block_reason=block_reason,
    )


def _row_haystack(row: Dict) -> str:
    return compact_text(
        "\n".join(
            str(row.get(key) or "")
            for key in (
                "doc_name",
                "heading",
                "section_title",
                "highlight_text",
                "article_id",
                "article_label",
                "clause_path",
                "parent_heading",
                "content",
                "retrieval_text",
                "body_text",
            )
        )
    )


def _row_matches_article(row: Dict, contract: RetrievalContract, haystack: str) -> bool:
    if not contract.article_variants:
        return True
    row_article = compact_text(str(row.get("article_id") or row.get("article_label") or row.get("structural_number") or ""))
    if row_article:
        return row_article in set(contract.article_variants)
    return any(variant and variant in haystack for variant in contract.article_variants)


def validate_evidence(contract: RetrievalContract, row: Dict) -> EvidenceValidation:
    if contract.block_reason:
        return EvidenceValidation(False, 0.0, (contract.block_reason,))

    coverage_policy = EvidenceCoveragePolicy.from_contract(contract)
    reasons: List[str] = []
    score = 0.0
    haystack = _row_haystack(row)
    doc_name = str(row.get("doc_name") or "")

    if contract.hard_target:
        allowed_docs = set(hard_target_doc_names(contract))
        if doc_name not in allowed_docs:
            return EvidenceValidation(False, 0.0, ("target_doc_mismatch",))
        score += 2.0

    if contract.article_variants:
        if _row_matches_article(row, contract, haystack):
            score += 2.0
        elif contract.hard_target:
            return EvidenceValidation(False, score, ("article_mismatch",))
        else:
            reasons.append("article_not_confirmed")
            score -= 0.8

    if contract.requires_table:
        source_type = str(row.get("source_type") or "").lower()
        role = str(row.get("chunk_role") or row.get("semantic_role") or "").lower()
        if "table" in source_type or role == "table" or any(token in haystack for token in ("表", "项目", "检查表", "检测项目")):
            score += 0.9
        else:
            reasons.append("table_evidence_not_confirmed")
            score -= 0.7

    matched_terms = tuple(
        term
        for term in contract.required_terms
        if term and any(compact_text(form) in haystack for form in code_query_variants(term))
    )
    missing_terms = tuple(
        term
        for term in contract.required_terms
        if term and not any(compact_text(form) in haystack for form in code_query_variants(term))
    )
    coverage = len(matched_terms) / max(len(contract.required_terms), 1)
    score += 1.4 * coverage
    if contract.intent == "definition_or_explanation":
        matched_focus = tuple(term for term in contract.focus_terms if term and _definition_focus_matches(term, row, haystack))
        missing_focus = tuple(term for term in contract.focus_terms if term and not _definition_focus_matches(term, row, haystack))
    else:
        matched_focus = tuple(term for term in contract.focus_terms if term and _term_matches(term, haystack))
        missing_focus = tuple(term for term in contract.focus_terms if term and not _term_matches(term, haystack))
    focus_coverage = len(matched_focus) / max(len(contract.focus_terms), 1)
    score += 0.9 * focus_coverage
    if matched_focus:
        reasons.append("focus_terms_covered")
    if matched_terms:
        reasons.append("required_terms_covered")
    if missing_terms or missing_focus:
        reasons.append("required_terms_missing")

    if contract.required_terms:
        if coverage_policy.strict_required and coverage < coverage_policy.required_threshold:
            return EvidenceValidation(False, score, tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)
        if not contract.hard_target and coverage < coverage_policy.required_threshold:
            return EvidenceValidation(False, score, tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)
    if coverage_policy.strict_focus and not contract.required_terms and contract.focus_terms and focus_coverage < coverage_policy.focus_threshold:
        if contract.targets and _any_term_matches(contract.focus_terms, _row_haystack({**row, "doc_name": ""})):
            reasons.append("focus_terms_not_confirmed")
        else:
            return EvidenceValidation(False, score, tuple(reasons), missing_focus, matched_focus)
    if coverage_policy.strict_focus and contract.required_terms and contract.focus_terms and focus_coverage <= 0.0:
        return EvidenceValidation(False, score, tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)
    if coverage_policy.all_focus_required and missing_focus and not contract.hard_target:
        return EvidenceValidation(False, score, tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)
    if contract.targets and not contract.hard_target and doc_name not in set(contract.synced_target_doc_names):
        if contract.focus_terms and focus_coverage <= 0.0:
            return EvidenceValidation(False, score, tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)

    return EvidenceValidation(True, round(score, 6), tuple(reasons), missing_terms + missing_focus, matched_terms + matched_focus)


def apply_contract(
    contract: RetrievalContract,
    rows: Iterable[Dict],
    *,
    keep_failed_when_no_hard_constraints: bool = False,
    preserve_order: bool = False,
) -> List[Dict]:
    passed: List[Tuple[float, Dict]] = []
    failed: List[Tuple[float, Dict]] = []

    for row in rows:
        validation = validate_evidence(contract, row)
        enriched = dict(row)
        enriched["contract_score"] = validation.score
        enriched["contract_passed"] = validation.passed
        enriched["contract_reasons"] = list(validation.reasons)
        enriched["contract_missing_terms"] = list(validation.missing_terms)
        enriched["contract_matched_terms"] = list(validation.matched_terms)
        total_score = validation.score + evidence_ranking_score(row)
        if validation.passed:
            passed.append((total_score, enriched))
        else:
            failed.append((total_score, enriched))

    if preserve_order:
        if passed:
            return [row for _, row in passed]
        if keep_failed_when_no_hard_constraints and not contract.has_hard_constraints:
            return [row for _, row in failed]
        return []

    passed.sort(key=lambda item: item[0], reverse=True)
    if passed:
        return [row for _, row in passed]
    if keep_failed_when_no_hard_constraints and not contract.has_hard_constraints:
        failed.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in failed]
    return []
