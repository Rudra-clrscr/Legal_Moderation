"""Deterministic corrections to LLM profile output, checked against the resume text itself.

A small local model reliably gets the gist of a resume but makes a few repeatable mistakes:
splitting a three-part name as first + last, copying the prompt's placeholder URL, skipping
links, and leaving ``currently_working_here`` false on "Jan 2021 - Present" roles. Each of these
facts is plainly visible in the text, so we fix them here without guessing: every correction
needs direct evidence in ``resume_text``.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

from app.profile_schemas import (
    AUTOFILL_MAX_SECONDARY_SKILLS,
    EXP_EMPLOYMENT_OPTIONS,
    KEY_OUTCOMES_MAX_CHARS,
    BONCProfileExtraction,
    EducationItem,
    LanguageItem,
    Skills,
    SocialLink,
    _match_enum,
    platform_from_url,
)

_MAX_SOCIAL_LINKS = 10
_NAME_SCAN_LINES = 15
_ENTRY_WINDOW_CHARS = 300

# Profile sites worth adding even when the model skipped them. Company or project URLs in
# experience descriptions are deliberately not added.
_PROFILE_DOMAINS = (
    "linkedin.com", "github.com", "gitlab.com", "instagram.com", "facebook.com", "twitter.com",
    "x.com", "dribbble.com", "behance.net", "medium.com", "stackoverflow.com", "kaggle.com",
    "leetcode.com",
)
_SCHEME_URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{},;|]+", re.IGNORECASE)
_BARE_PROFILE_RE = re.compile(
    r"(?<![@\w.\-/])(?:www\.)?(?:" + "|".join(re.escape(d) for d in _PROFILE_DOMAINS) + r")/[^\s<>\"'()\[\]{},;|]+",
    re.IGNORECASE,
)
_NAME_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]*\.?$")
# A date range "<year> - <end>": the end is a date, a month name, or an ongoing word.
_DATE_RANGE_RE = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:-|–|—|to|till)\s*(\d{1,2}/(?:19|20)\d{2}|(?:19|20)\d{2}|[A-Za-z]+(?:\s+date)?)",
    re.IGNORECASE,
)
_ONGOING_ENDS = {"present", "current", "currently", "now", "ongoing", "today", "till date", "to date"}


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def _display_case(word: str) -> str:
    return word.capitalize() if word.isupper() and len(word) > 1 else word


def _find_name_line(text: str, first_name: str) -> list[str] | None:
    """Words of the first short heading-like line that starts with ``first_name``."""
    target = first_name.strip().lower()
    seen = 0
    for raw in text.splitlines():
        line = raw.strip().strip("#*").strip()
        if not line:
            continue
        seen += 1
        if seen > _NAME_SCAN_LINES:
            break
        words = line.split()
        if not 2 <= len(words) <= 4 or len(line) > 40:
            continue
        if not all(_NAME_WORD_RE.match(w) for w in words):
            continue
        if words[0].rstrip(".").lower() == target:
            return words
    return None


def _ground_name(profile: BONCProfileExtraction, text: str) -> None:
    b = profile.basic_details
    if not b.first_name or " " in b.first_name.strip():
        return
    words = _find_name_line(text, b.first_name)
    if not words:
        return
    lowered = [w.lower().rstrip(".") for w in words]
    # Only correct a split the line can explain: the model's last/middle name must be in it
    # (a middle initial such as "A" counts for "Anand").
    if b.last_name and any(p.lower() not in lowered for p in b.last_name.split()):
        return
    # With the last name confirmed by the heading line, the line is the full name: a middle
    # name it lacks came from elsewhere (an email "himani_jha@..." gave "Jha").
    if b.middle_name and not b.last_name:
        for p in b.middle_name.split():
            p = p.lower().rstrip(".")
            if not p or not any(w == p or (len(p) == 1 and w.startswith(p)) for w in lowered[1:]):
                return
    b.first_name = _display_case(words[0])
    b.last_name = _display_case(words[-1])
    b.middle_name = " ".join(_display_case(w) for w in words[1:-1]) or None


# ---------------------------------------------------------------------------
# Social links
# ---------------------------------------------------------------------------
def _link_key(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}".lower()


def _is_real_host(url: str) -> bool:
    host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
    labels = host.split(".")
    return len(labels) >= 2 and all(labels) and re.fullmatch(r"[a-z]{2,}", labels[-1].lower()) is not None


def _urls_in_text(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for m in _SCHEME_URL_RE.finditer(text):
        found.append((m.start(), m.group(0)))
    for m in _BARE_PROFILE_RE.finditer(text):
        found.append((m.start(), f"https://{m.group(0)}"))
    out: list[str] = []
    seen: set[str] = set()
    for _, url in sorted(found):
        url = url.rstrip(".:")
        key = _link_key(url)
        if _is_real_host(url) and key not in seen:
            seen.add(key)
            out.append(url)
    return out


def _ground_links(profile: BONCProfileExtraction, text: str) -> None:
    text_urls = _urls_in_text(text)
    by_key = {_link_key(u): u for u in text_urls}
    squashed_text = _squash(text)
    links: list[SocialLink] = []
    seen: set[str] = set()

    model_keys = {_link_key(link.url) for link in profile.networking.social_links}
    for link in profile.networking.social_links:
        url = link.url
        if "@" in urlparse(url if "://" in url else f"https://{url}").netloc:
            continue  # an email address the model wrote as a link ("https://name@example.com")
        host = _link_key(url).split("/")[0]
        if _is_real_host(url) and _link_key(url) not in by_key:
            # Not in the resume (often a misspelt path): take the resume's link on the same site.
            same_site = [u for k, u in by_key.items() if k.split("/")[0] == host and k not in model_keys]
            if len(same_site) == 1:
                url = same_site[0]
        if not _is_real_host(url):
            # e.g. "https://.../rohank-example": recover the real URL by its path, else drop it.
            tail = urlparse(url if "://" in url else f"https://{url}").path.rstrip("/").lower()
            match = next((u for u in text_urls if tail and _link_key(u).endswith(tail)), None)
            if match is None:
                continue
            url = match
        elif _link_key(url) in by_key:
            url = by_key[_link_key(url)]  # the resume's exact spelling
        elif _squash(_link_key(url)) not in squashed_text:
            # Invented from the name ("linkedin.com/in/<first>-<last>", "<name>.com") when the
            # resume only has an email address: never fill a link the resume does not contain.
            continue
        key = _link_key(url)
        if key in seen:
            continue
        seen.add(key)
        links.append(SocialLink(platform=link.platform, url=url))

    for url in text_urls:
        key = _link_key(url)
        host = key.split("/")[0]
        if key in seen or not any(host == d or host.endswith("." + d) for d in _PROFILE_DOMAINS):
            continue
        if "/" not in key:  # bare domain such as "linkedin.com" is not a profile
            continue
        seen.add(key)
        links.append(SocialLink(platform=platform_from_url(url) or "Website", url=url))

    profile.networking.social_links = links[:_MAX_SOCIAL_LINKS]


# ---------------------------------------------------------------------------
# Ongoing roles / studies
# ---------------------------------------------------------------------------
def _entry_window(text_lower: str, name: str, other_names: list[str], size: int = _ENTRY_WINDOW_CHARS) -> str | None:
    start = text_lower.find(name.lower())
    if start < 0:
        return None
    end = min(len(text_lower), start + len(name) + size)
    for other in other_names:
        pos = text_lower.find(other.lower(), start + len(name))
        if 0 <= pos < end:
            end = pos
    return text_lower[start:end]


def _ground_ongoing(items: list, name_attr: str, flag_attr: str, text: str, all_names: list[str]) -> None:
    """Flag entries whose own date range (the first one after the entry's name) ends in "Present"."""
    text_lower = text.lower()
    for item in items:
        name = getattr(item, name_attr)
        if not name or item.to_date or getattr(item, flag_attr):
            continue
        window = _entry_window(text_lower, name, [n for n in all_names if n.lower() != name.lower()])
        first_range = _DATE_RANGE_RE.search(window) if window else None
        if first_range and first_range.group(1).lower() in _ONGOING_ENDS:
            setattr(item, flag_attr, True)


# ---------------------------------------------------------------------------
# Skills, tools, languages, certifications: keep only what the resume mentions
# ---------------------------------------------------------------------------
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_PROGRAMMING_LANGUAGES = {
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go", "golang", "rust", "ruby", "php",
    "kotlin", "swift", "scala", "sql", "r", "perl", "dart", "html", "css", "bash", "matlab",
}


def _squash(text: str) -> str:
    return _NON_ALNUM_RE.sub("", text.lower())


def _mentioned(name: str, squashed_text: str) -> bool:
    key = _squash(name)
    if len(key) < 3:  # "C", "R", "Go": too short to check reliably, keep
        return True
    if key in squashed_text:
        return True
    words = [w for w in _NON_ALNUM_RE.split(name.lower()) if len(w) >= 4]
    return bool(words) and sum(w in squashed_text for w in words) * 2 >= len(words)


_NO_EXPIRY_RE = re.compile(r"\b(?:no expiry|no expiration|does not expire|never expires|lifetime|non-expiring)\b", re.IGNORECASE)


def _merge_split_names(names: list[str], text_lower: str) -> list[str]:
    """Rejoin a term the model split in two ("Agentic AI" + "Workflows") when the resume
    writes it as one phrase. Neighbours split by a comma stay apart, and so do one-per-line
    lists; a line break only counts as a space inside a wrapped comma-separated line."""
    out: list[str] = []
    for name in names:
        if out:
            joined = f"{out[-1]} {name}"
            pattern = r"(?<![a-z0-9])" + r"[ \t]*\n?[ \t]*".join(map(re.escape, joined.lower().split())) + r"(?![a-z0-9])"
            for m in re.finditer(pattern, text_lower):
                if "\n" not in m.group(0) or _is_wrapped_list_line(text_lower, m.start()):
                    out[-1] = joined
                    break
            else:
                out.append(name)
            continue
        out.append(name)
    return out


_WRAPPED_LINE_MIN_CHARS = 60  # a line only wraps once it reaches the page width


def _is_wrapped_list_line(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    line = text[line_start: text.find("\n", pos)].rstrip()
    return ", " in line and not line.endswith(",") and len(line) >= _WRAPPED_LINE_MIN_CHARS


def _merge_split_skills(profile: BONCProfileExtraction, text_lower: str) -> None:
    skills = profile.education_and_experience.skills
    tiers = [(s, "primary") for s in skills.primary] + [(s, "secondary") for s in skills.secondary]
    merged = _merge_split_names([s.name for s, _ in tiers], text_lower)
    if len(merged) == len(tiers):
        return
    by_name = {s.name: (s, tier) for s, tier in tiers}
    primary, secondary = [], []
    for name in merged:
        original = by_name.get(name)
        if original is None:  # a joined name: keep the first part's rating and tier
            first = next(s for s, _ in tiers if name.startswith(s.name + " "))
            original = (first.model_copy(update={"name": name}), by_name[first.name][1])
        (primary if original[1] == "primary" else secondary).append(original[0])
    skills.primary, skills.secondary = primary, secondary


_PAREN_RE = re.compile(r"\(([^()]*)\)")


def _cert_lines(cert, text: str) -> str | None:
    """The certificate's line in the resume, plus the next one only when the name wrapped onto it
    (the next line is usually another certificate, with its own date)."""
    lines = text.splitlines()
    low = [ln.lower() for ln in lines]
    name, issuer = (cert.name or "").lower(), (cert.issuer or "").lower()
    # Name and issuer together first: a short name ("Python") also appears in the skills list.
    tests = []
    if name and issuer:
        tests.append(lambda ln: name in ln and issuer in ln)
    if len(name) >= 4:
        tests.append(lambda ln: name in ln)
    if len(issuer) >= 4:
        tests.append(lambda ln: issuer in ln)
    for test in tests:
        for i, ln in enumerate(low):
            if test(ln):
                return lines[i] if name in ln else " ".join(lines[i: i + 2])
    return None


def _ground_cert_date(cert, text: str) -> None:
    """An issue date needs its year on the certificate's own line: the model copies one
    certificate's date onto the next. A "(Jan 2024 - Mar 2024)" course was issued at its end."""
    own = _cert_lines(cert, text)
    if own is None:
        return
    period = next((p for p in _PAREN_RE.findall(own) if _YEAR_RE.search(p)), None)
    dated = [d.strip() for d in _PERIOD_SPLIT_RE.split(period) if _YEAR_RE.search(d)] if period else []
    if cert.issue_date and not set(_YEAR_RE.findall(cert.issue_date)) & set(_YEAR_RE.findall(own)):
        cert.issue_date = None
    if dated and (not cert.issue_date or (len(dated) >= 2 and _squash(cert.issue_date) == _squash(dated[0]))):
        cert.issue_date = dated[-1]


def _ground_certifications(profile: BONCProfileExtraction, text: str) -> None:
    for cert in profile.education_and_experience.certifications:
        _split_cert_provider(cert)
        # "PMP - Project Management Institute": the issuer belongs in its own field.
        if cert.issuer:
            for sep in (" - ", " – ", " — ", ", "):
                suffix = f"{sep}{cert.issuer}"
                if cert.name.lower().endswith(suffix.lower()) and len(cert.name) > len(suffix):
                    cert.name = cert.name[: -len(suffix)].strip()
        _ground_cert_date(cert, text)
        if cert.does_not_expire or cert.expiry_date:
            continue
        pos = text.lower().find(cert.name.lower())
        if pos >= 0:
            line_end = text.find("\n", pos)
            if _NO_EXPIRY_RE.search(text[pos: line_end if line_end >= 0 else len(text)]):
                cert.does_not_expire = True


def _ground_lists(profile: BONCProfileExtraction, text: str) -> None:
    squashed = _squash(text)
    _merge_split_skills(profile, text.lower())
    _ground_certifications(profile, text)
    ee = profile.education_and_experience
    ee.skills.primary = [s for s in ee.skills.primary if _mentioned(s.name, squashed)]
    ee.skills.secondary = [s for s in ee.skills.secondary if _mentioned(s.name, squashed)]
    # Dropped primary skills leave room: promote from secondary to keep the top five full.
    while len(ee.skills.primary) < 5 and ee.skills.secondary:
        ee.skills.primary.append(ee.skills.secondary.pop(0))
    ee.tools = [t for t in ee.tools if _mentioned(t, squashed)]
    ee.languages = [
        lang for lang in ee.languages
        if lang.language.lower() not in _PROGRAMMING_LANGUAGES and _mentioned(lang.language, squashed)
    ]
    ee.certifications = [c for c in ee.certifications if _mentioned(c.name, squashed)]
    prefs = profile.career_preferences
    prefs.target_roles = [r for r in prefs.target_roles if _mentioned(r, squashed)]
    prefs.preferred_locations = [loc for loc in prefs.preferred_locations if _mentioned(loc, squashed)]
    ee.sub_domains = [d for d in ee.sub_domains if _mentioned(d, squashed)]
    _fill_from_sections(profile, text)
    _move_tools_out_of_skills(profile)
    _ground_domains(profile)


# ---------------------------------------------------------------------------
# Fallback: read skills, tools and spoken languages from the resume's own sections
# ---------------------------------------------------------------------------
# The model sometimes skips these lists (long resumes, or it invents its own keys). The
# resume's "Technical Skills" / "Languages Known" sections hold them in a predictable form.
_LIST_BULLET_RE = re.compile(r"^\s*#*\s*(?:[•\-*–·▪●◦➢►✓]+|o(?=\s))?\s*")  # "## " from OCR markdown
_SKILLS_HEAD_RE = re.compile(
    r"\b(?:skills?|skill\s*set|competenc(?:y|ies)|expertise|tech(?:nical)?\s+stack|technologies)\b", re.IGNORECASE
)
_LANG_HEAD_RE = re.compile(r"^(?:languages?\s+known|known\s+languages|spoken\s+languages|languages?)\b", re.IGNORECASE)
_OTHER_HEAD_RE = re.compile(
    r"^(?:work\s+)?(?:experience|employment|education|academics?|projects?|certifications?|certificates?|achievements?"
    r"|awards?|hobbies|interests|personal|declaration|summary|profile|objective|publications?|references?"
    r"|internships?|training|activities|strengths|contact|career)\b",
    re.IGNORECASE,
)
_TOOL_LABEL_RE = re.compile(
    r"tool|framework|technolog|platform|database|\bdb\b|software|\bide\b|cloud|devops|librar|server|operating|\bos\b",
    re.IGNORECASE,
)
_DOMAIN_LABEL_RE = re.compile(r"domain|industr|sector", re.IGNORECASE)
_BARE_LABEL_RE = re.compile(
    r"^(?:languages?|tools?|databases?|frameworks?|technologies|cloud|platforms?|libraries|devops|ides?)\s+(?=\S)",
    re.IGNORECASE,
)
_ITEM_SPLIT_RE = re.compile(r"\s*(?:[,;|]|\s•\s|\s/\s)\s*")
_DATE_LIKE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{2,4}\b")
_SECTION_MAX_LINES = 20
_HEAD_MAX_WORDS = 4
_ITEM_MAX_CHARS = 40
_ITEM_MAX_WORDS = 4
_ACTION_VERBS = {
    "delivered", "resolved", "created", "developed", "built", "designed", "implemented", "managed", "led", "worked",
    "improved", "reduced", "increased", "collaborated", "maintained", "provided", "integrated", "optimized",
    "handled", "responsible", "performed", "achieved", "ensured", "supported", "using", "used",
}
_SPOKEN_LANGUAGES = {
    "english", "hindi", "bengali", "bangla", "marathi", "tamil", "telugu", "kannada", "malayalam", "gujarati",
    "punjabi", "urdu", "odia", "oriya", "assamese", "konkani", "sindhi", "kashmiri", "nepali", "sanskrit",
    "maithili", "bhojpuri", "tulu", "french", "german", "spanish", "italian", "portuguese", "russian", "arabic",
    "persian", "farsi", "japanese", "korean", "chinese", "mandarin", "cantonese", "dutch", "swedish", "turkish",
    "thai", "vietnamese", "indonesian", "malay", "sinhala", "hebrew", "greek", "polish",
}
_LANG_ITEM_RE = re.compile(r"^([A-Za-z]+)\s*(?:[(\-–—:]\s*([A-Za-z /]+?)\s*\)?)?$")


def _is_heading(line: str) -> bool:
    return len(line) <= 40 and "," not in line and bool(_OTHER_HEAD_RE.match(line) or _LANG_HEAD_RE.match(line))


def _ends_section(line: str) -> bool:
    """A new heading, an all-caps title ("PROFESSIONAL EXPERIENCE") or a dated line."""
    letters = [c for c in line if c.isalpha()]
    all_caps = len(letters) >= 4 and line.upper() == line and "," not in line and len(line) <= 40
    return _is_heading(line) or all_caps or bool(_DATE_LIKE_RE.search(line))


def _split_items(values: str, strict: bool = False) -> list[str]:
    """Split a comma list into short items. ``strict`` (unlabelled lines) rejects the whole
    line when it reads like a sentence that merely contains commas."""
    raw_items = [re.sub(r"^(?:and|&)\s+", "", r.strip().rstrip(".")).strip() for r in _split_outside_brackets(values)]
    raw_items = [r for r in raw_items if r]
    if strict and (values.rstrip().endswith(".") or any(len(r.split()) > _ITEM_MAX_WORDS for r in raw_items)):
        return []
    out = []
    for item in raw_items:
        if not (2 <= len(item) <= _ITEM_MAX_CHARS and len(item.split()) <= _ITEM_MAX_WORDS):
            continue
        if not item[0].isalnum() or item.endswith(":") or _DATE_LIKE_RE.search(item) or not re.search(r"[A-Za-z]", item):
            continue
        if re.search(r"\s[—–]\s", item) or item.split()[0].lower() in _ACTION_VERBS:  # "Course — Coursera", "Delivered secure"
            continue
        if item == item.lower() and item[0].isalpha():  # "fixed bugs", "and": prose, not a skill name
            continue
        out.append(item)
    return out


def _split_outside_brackets(values: str) -> list[str]:
    """Split on list separators, but keep "LLM APIs (OpenAI, LangChain)" in one piece."""
    parts, depth, start = [], 0, 0
    for m in re.finditer(r"[()]|" + _ITEM_SPLIT_RE.pattern, values):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            parts.append(values[start:m.start()])
            start = m.end()
    parts.append(values[start:])
    return [x for x in parts if x.strip()]


def _is_sub_label(lines: list[str], k: int) -> bool:
    """A short title-case label alone on its line ("Frameworks", "Languages & Data") whose
    comma list starts on the next line. A real "Languages" section (English, Hindi) is not one."""
    line = lines[k]
    if (not line or "," in line or ":" in line or len(line) > 40 or len(line.split()) > _HEAD_MAX_WORDS
            or line.upper() == line or _DATE_LIKE_RE.search(line) or _OTHER_HEAD_RE.match(line)):
        return False
    nxt = next((ln for ln in lines[k + 1: k + 3] if ln), "")
    if "," not in nxt or ":" in nxt:
        return False
    return not any(w.lower() in _SPOKEN_LANGUAGES for w in re.findall(r"[A-Za-z]+", nxt))


def _is_skills_heading(label: str) -> bool:
    return (
        len(label.split()) <= _HEAD_MAX_WORDS and "," not in label and not re.search(r"\s[—–-]\s", label)
        and bool(_SKILLS_HEAD_RE.search(label)) and not _OTHER_HEAD_RE.match(label)
    )


def _section_lists(text: str) -> tuple[list[str], list[str], list[str]]:
    """Skills, tools and domains listed under skills-type headings. Labelled lines like
    "Developer Tools: Git, IIS" go to tools, "Domains: ..." to domains, others to skills."""
    found: dict[str, list[str]] = {"skills": [], "tools": [], "domains": []}

    def route(label: str, default: str) -> str:
        if _DOMAIN_LABEL_RE.search(label):
            return "domains"
        if _TOOL_LABEL_RE.search(label) and not re.search(r"skill", label, re.IGNORECASE):
            return "tools"
        return default

    lines = [_LIST_BULLET_RE.sub("", ln).strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        head = lines[i]
        label, _, inline = head.partition(":")
        if not _is_skills_heading(label):
            i += 1
            continue
        target = route(label, "skills")
        j = i + 1
        if inline.strip():
            # "Technologies: Java, Spring" (often inside a project): just that line and its wrap.
            body = [head]
            while j < len(lines) and body[-1].rstrip().endswith(","):
                body.append(lines[j])
                j += 1
        else:
            body, filled = [], 0
            while j < len(lines) and filled < _SECTION_MAX_LINES and (
                _is_sub_label(lines, j) or not _ends_section(lines[j])
            ):
                body.append(lines[j])
                filled += bool(lines[j])
                j += 1
        current, continues, labelled = target, False, False
        for k, line in enumerate(body):
            if not line:
                continues = labelled = False
                continue
            if _is_sub_label(body, k):  # "Frameworks" on its own line, its list on the next
                target = current = route(line, "skills")
                labelled = True
                continue
            sub_label, sep, values = line.partition(":")
            if sep and len(sub_label) <= 40 and "," not in sub_label:
                current = route(sub_label, "skills")
                if not values.strip():  # "Domains:" with its values on the following lines
                    target = current
                items = _split_items(values)
            elif continues or labelled:
                items = _split_items(line)
            else:
                bare = _BARE_LABEL_RE.match(line)  # "Tools Git, TFS": a label that lost its colon
                if bare:
                    current = route(bare.group(0), "skills")
                    items = _split_items(line[bare.end():])
                else:
                    current = target
                    items = _split_items(line, strict=True) if ("," in line or len(line.split()) <= _ITEM_MAX_WORDS) else []
            found[current].extend(items)
            continues = line.rstrip().endswith(",")
        i = j
    return found["skills"], found["tools"], found["domains"]


def _section_languages(text: str) -> list[dict[str, str | None]]:
    found: dict[str, dict[str, str | None]] = {}
    lines = [_LIST_BULLET_RE.sub("", ln).strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        m = _LANG_HEAD_RE.match(line)
        if not m or len(m.group(0)) > 30:
            continue
        rest = line[m.end():].lstrip(" :-–")
        body = ([rest] if rest else []) + lines[i + 1: i + 7]
        for chunk in body:
            if not rest and _is_heading(chunk):
                break
            for item in re.split(r"\s*(?:[,;|/]|\band\b)\s*", chunk):
                lm = _LANG_ITEM_RE.match(item.strip())
                if lm and lm.group(1).lower() in _SPOKEN_LANGUAGES:
                    found.setdefault(lm.group(1).lower(), {"language": lm.group(1).title(), "proficiency": lm.group(2)})
            if rest:  # an inline list ends on its own line
                break
    return list(found.values())


def _profile_names(profile: BONCProfileExtraction) -> tuple[set[str], set[str]]:
    """Organisation names, and title/place names, from the rest of the profile. A two-column
    layout can interleave them with the skills list, and they are never skills."""
    ee, b = profile.education_and_experience, profile.basic_details
    orgs = {_squash(n) for n in [e.company_name for e in ee.experience] + [e.institute_name for e in ee.education] if n}
    others = [e.designation for e in ee.experience] + [e.degree for e in ee.education]
    places = [e.location for e in ee.experience + ee.education]
    places += [b.primary_address.city, b.primary_address.state, b.primary_address.country, profile.networking.location]
    exact = {_squash(n) for n in others if n} | {_squash(part) for p in places if p for part in p.split(",")}
    return {o for o in orgs if len(o) >= 5}, exact - {""}


_SPARSE_SKILLS = 5


def _fill_from_sections(profile: BONCProfileExtraction, text: str) -> None:
    ee = profile.education_and_experience
    orgs, exact = _profile_names(profile)

    def keep(item: str) -> bool:
        key = _squash(item)
        return key not in exact and not any(o in key for o in orgs)

    skills, tools, domains = ([x for x in found if keep(x)] for found in _section_lists(text))
    current = ee.skills.primary + ee.skills.secondary
    if len(current) < _SPARSE_SKILLS and skills:
        # The model returned few or no skills: top up from the resume's own skills section.
        have = [_squash(s.name) for s in current]
        # Skip pieces of a skill already listed ("Agentic AI" of a wrapped "Agentic AI Workflows").
        extra = [x for x in skills if not any(_squash(x) in h for h in have)]
        ee.skills = Skills.model_validate([s.model_dump() for s in current] + extra)
    taken = {_squash(s.name) for s in ee.skills.primary + ee.skills.secondary}
    if not ee.tools and tools:
        ee.tools = _dedupe([t for t in tools if _squash(t) not in taken])[:30]
    if not ee.sub_domains and domains:
        ee.sub_domains = _dedupe(domains)[:10]
    if not ee.languages:
        ee.languages = [LanguageItem.model_validate(item) for item in _section_languages(text)][:10]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (_squash(x) in seen or seen.add(_squash(x)))]


# Frameworks, software, platforms and methodologies: these belong under Tools, not Skills.
_TOOL_KEYS = {
    # frameworks and libraries
    "net", "netcore", "netframework", "aspnet", "aspnetmvc", "aspnetcore", "aspnetcoremvc", "aspnetwebapi",
    "aspnetwebforms", "netcoreapi", "adonet", "entityframework", "entityframeworkcore",
    "webapi", "jquery", "react", "reactjs", "reactnative", "angular", "angularjs", "vue", "vuejs", "nextjs",
    "nodejs", "expressjs", "django", "flask", "fastapi", "spring", "springboot", "hibernate", "laravel",
    "rails", "rubyonrails", "bootstrap", "tailwind", "tailwindcss", "redux", "flutter", "tensorflow",
    "pytorch", "keras", "pandas", "numpy", "scikitlearn", "opencv", "langchain", "selenium", "junit", "pytest",
    # software and platforms
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "docker", "kubernetes", "jenkins",
    "terraform", "ansible", "postman", "swagger", "visualstudio", "vscode", "visualstudiocode", "eclipse",
    "intellij", "androidstudio", "xcode", "excel", "msexcel", "advancedexcel", "msoffice", "microsoftoffice",
    "powerbi", "tableau", "sap", "tally", "tallyerp", "salesforce", "figma", "photoshop", "illustrator",
    "autocad", "solidworks", "catia", "sqlserver", "mssqlserver", "mysql", "postgresql", "postgres",
    "mongodb", "oracle", "firebase", "redis", "elasticsearch", "aws", "azure", "gcp", "googlecloud", "linux",
    "unix", "windowsserver", "iis", "wordpress", "shopify", "unity", "blender", "zoho", "hubspot",
    "sharepoint", "servicenow", "crystalreports", "ssrs", "ssis", "kafka", "rabbitmq", "nginx",
    # methodologies
    "agile", "scrum", "kanban", "sixsigma", "leansixsigma", "itil", "pmp",
}


def _is_tool(name: str) -> bool:
    key = _squash(name)
    return key in _TOOL_KEYS or key.rstrip("0123456789") in _TOOL_KEYS  # "Angular 14", "Tally ERP 9"


def _move_tools_out_of_skills(profile: BONCProfileExtraction) -> None:
    """A resume's single SKILLS list mixes languages with frameworks and software, and the model
    copies it all into skills. When it gave no tools, move the secondary skills that are tools."""
    ee = profile.education_and_experience
    if ee.tools and not (ee.skills.primary or ee.skills.secondary):
        # The opposite slip: a "Tools/Technologies:" list read whole into tools leaves no skills.
        # Languages and anything not a known tool ("C#", "JavaScript") are skills.
        to_skills = [t for t in ee.tools if t.lower() in _PROGRAMMING_LANGUAGES or not _is_tool(t)]
        if to_skills:
            ee.skills = Skills.model_validate(to_skills)
            ee.tools = [t for t in ee.tools if t not in to_skills]
        return
    if ee.tools:
        return
    moved = [s.name for s in ee.skills.secondary if _is_tool(s.name)]
    if moved:
        ee.skills.secondary = [s for s in ee.skills.secondary if not _is_tool(s.name)]
        ee.tools = moved


_SOFTWARE_ROLE_RE = re.compile(
    r"developer|programmer|software|devops|data\s+scien|full[\s-]?stack|front[\s-]?end|back[\s-]?end|\bsde\b|\.net",
    re.IGNORECASE,
)
_MIN_TECH_ITEMS = 3


def _ground_domains(profile: BONCProfileExtraction) -> None:
    """Sub-domains are niches, not a second skills list ("Python", "SQL" are dropped). Long
    resumes can run out of output before the industry; a software role with a technical
    skill set is in Technology & Software."""
    ee = profile.education_and_experience
    skills = ee.skills.primary + ee.skills.secondary
    names = {_squash(s.name) for s in skills} | {_squash(t) for t in ee.tools}
    ee.sub_domains = [
        d for d in ee.sub_domains
        if _squash(d) not in names and not _is_tool(d) and d.lower() not in _PROGRAMMING_LANGUAGES
    ]
    if ee.industry_clusters:
        return
    roles = [e.designation or "" for e in ee.experience] + [profile.networking.profession or ""]
    tech = [n for n in [s.name for s in skills] + ee.tools if _is_tool(n) or n.lower() in _PROGRAMMING_LANGUAGES]
    if any(_SOFTWARE_ROLE_RE.search(r) for r in roles) and len(tech) >= _MIN_TECH_ITEMS:
        ee.industry_clusters = ["Technology & Software"]


# "SSM Learning Centre (Jan 2024 - Mar 2024) - .Net Full Stack Development": provider, period, course.
_CERT_PERIOD_RE = re.compile(r"^(?P<left>[^()]+?)\s*\((?P<period>[^()]*)\)\s*[-–—:]\s*(?P<right>.+)$")
_DURATION_RE = re.compile(r"^\d+\s*(?:days?|weeks?|months?|years?|hours?|hrs?)$", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PERIOD_SPLIT_RE = re.compile(r"\s*(?:-|–|—|\bto\b)\s*", re.IGNORECASE)
_COURSE_WORD_RE = re.compile(
    r"\b(?:certifi\w*|course|training|program+e?|diploma|bootcamp|workshop|associate|professional|"
    r"practitioner|specialist|developer|architect|engineer|analyst|fundamentals|essentials)\b",
    re.IGNORECASE,
)


def _split_cert_provider(cert) -> None:
    m = _CERT_PERIOD_RE.match(cert.name)
    if not m:
        return
    period = m.group("period").strip()
    dated = [p for p in _PERIOD_SPLIT_RE.split(period) if _YEAR_RE.search(p)]
    if not (dated or _DURATION_RE.match(period)):  # "(During College)" is not a period
        return
    left, right = m.group("left").strip(), m.group("right").strip()
    if cert.issuer:
        # The model may fill the issuer and still leave the whole line in the name.
        issuer = _squash(cert.issuer)
        if _squash(left) == issuer:
            cert.name = right
        elif _squash(right) == issuer:
            cert.name = left
        else:
            return  # a different issuer: leave the model's reading alone
    elif _COURSE_WORD_RE.search(left) and not _COURSE_WORD_RE.search(right):
        cert.name, cert.issuer = left, right  # "AWS Developer Associate (2023) - Amazon"
    else:
        cert.name, cert.issuer = right, left
    if dated and not cert.issue_date:
        cert.issue_date = dated[-1]  # completed at the end of the period


# ---------------------------------------------------------------------------
# Details the model copies from elsewhere in the resume
# ---------------------------------------------------------------------------
_SCHOOL_DEGREE_RE = re.compile(
    r"^(?:10th|12th|10\s*\+\s*2|xii?|ssc|hsc|sslc|hslc|matric\w*|(?:senior\s+|higher\s+)?secondary|"
    r"intermediate|class\s+(?:xii?|1[02]))\b",
    re.IGNORECASE,
)
_LANG_LEVEL_RE = re.compile(
    r"\b(?:native|bilingual|mother\s*tongue|fluent\w*|professional|conversational|elementary|basic|beginner|"
    r"intermediate|advanced|proficient|expert|working|read|write|speak)\b",
    re.IGNORECASE,
)
_EMPLOYMENT_WORDS = (
    ("Internship", re.compile(r"\binternship\b|\bintern\b", re.IGNORECASE)),
    ("Part-time", re.compile(r"\bpart[\s-]?time\b", re.IGNORECASE)),
    ("Contract", re.compile(r"\bcontract(?:ual)?\b", re.IGNORECASE)),
    ("Full-time", re.compile(r"\bfull[\s-]?time\b", re.IGNORECASE)),
)
_REMOTE_RE = re.compile(r"\bremote\b", re.IGNORECASE)
_CONTEXT_LINES = 3
_PLACE_WORD_RE = re.compile(r"[^a-z]+")


def _own_line(text_lower: str, name: str) -> str | None:
    pos = text_lower.find(name.lower())
    if pos < 0:
        return None
    end = text_lower.find("\n", pos)
    return text_lower[text_lower.rfind("\n", 0, pos) + 1: end if end >= 0 else len(text_lower)]


def _place_words(place: str | None) -> set[str]:
    return {w for w in _PLACE_WORD_RE.split((place or "").lower()) if len(w) >= 3}


def _ground_entry_details(profile: BONCProfileExtraction, text: str, all_names: list[str]) -> None:
    ee = profile.education_and_experience
    text_lower = text.lower()
    lines = text_lower.splitlines()
    addr = profile.basic_details.primary_address
    home = _place_words(addr.city) | _place_words(addr.state) | _place_words(profile.networking.location)

    raw = text.splitlines()

    def context(name: str) -> str:
        """Each line naming the entry plus the few lines under it, up to the next entry or heading
        (a name can also appear in a summary, so every mention counts)."""
        key = name.lower()
        others = [n.lower() for n in all_names if n.lower() != key]
        out: list[str] = []
        for idx in (i for i, ln in enumerate(lines) if key in ln):
            out.append(lines[idx])
            for i in range(idx + 1, min(len(lines), idx + 1 + _CONTEXT_LINES)):
                head = raw[i].strip()
                if key in lines[i] or any(o in lines[i] for o in others) or (
                    head and head.upper() == head and any(c.isalpha() for c in head)
                ):
                    break
                out.append(lines[i])
        return "\n".join(out)

    def home_copy(place: str | None, name: str, ctx: str) -> bool:
        # The candidate's own city given to an entry whose part of the resume never names it
        # (the entry's own name does not count: "Maharashtra Board" is not in Maharashtra's text).
        words = _place_words(place)
        ctx = ctx.replace(name.lower(), " ")
        return bool(words & home) and not any(w in ctx for w in words)

    for edu in ee.education:
        if not edu.institute_name or text_lower.find(edu.institute_name.lower()) < 0:
            continue
        ctx = context(edu.institute_name)
        if home_copy(edu.location, edu.institute_name, ctx):
            edu.location = None
        # "10+2 ... (2004)": a stream belongs to a school entry only if its own line names it.
        if edu.field_of_study and edu.degree and _SCHOOL_DEGREE_RE.match(edu.degree.strip()):
            hits = [i for i, ln in enumerate(lines) if edu.degree.lower() in ln]
            field = _squash(edu.field_of_study)
            if hits and not any(field in _squash(" ".join(lines[i:i + 2])) for i in hits):
                edu.field_of_study = None
        # A lone year is the passing year, not the start.
        if edu.from_date and not edu.to_date and not edu.currently_studying_here and not _DATE_RANGE_RE.search(ctx):
            edu.from_date, edu.to_date = None, edu.from_date

    for exp in ee.experience:
        if not exp.company_name or text_lower.find(exp.company_name.lower()) < 0:
            continue
        ctx = context(exp.company_name)
        line = _own_line(text_lower, exp.company_name) or ""
        if home_copy(exp.location, exp.company_name, ctx):
            exp.location = None
        if not exp.location and _REMOTE_RE.search(line):
            exp.location = "Remote"
        # The role's own words win over the model's default ("Full-time" on an "Intern" role).
        own = f"{line} {exp.designation or ''}"
        stated = next((t for t, rx in _EMPLOYMENT_WORDS if rx.search(own)), None)
        if stated:
            exp.employment_type = stated

    # A proficiency the resume never states is the model's guess: leave it for the user.
    raw_lines = text.splitlines()
    for lang in ee.languages:
        if not lang.proficiency:
            continue
        name_re = re.compile(rf"\b{re.escape(lang.language)}\b", re.IGNORECASE)
        hits = [(i, m) for i, ln in enumerate(raw_lines) for m in name_re.finditer(ln)]
        if hits and not any(_states_level(raw_lines, i, m) for i, m in hits):
            lang.proficiency = None


_LIST_SEP_RE = re.compile(r"[,;|•]")


def _states_level(lines: list[str], i: int, m: re.Match) -> bool:
    """A level in this language's own list item: "English (Fluent)", "Fluent in English", or a
    table row with the level on the next line. "Hindi, English (Fluent)" gives Hindi no level."""
    line = lines[i]
    before = _LIST_SEP_RE.split(line[: m.start()])[-1]
    rest = line[m.end():]
    if not rest.strip():  # the language ends the line: its level may sit on the next one
        rest = lines[i + 1] if i + 1 < len(lines) else ""
    after = _LIST_SEP_RE.split(rest)[0]
    return bool(_LANG_LEVEL_RE.search(before) or _LANG_LEVEL_RE.search(after))


_OUTCOME_WINDOW_CHARS = 1500
_OUTCOME_MIN_WORD_SHARE = 0.7
_BULLET_PREFIX_RE = re.compile(r"^\s*[•\-*–·]+\s*")


def _ground_outcomes(profile: BONCProfileExtraction, text: str, all_names: list[str]) -> None:
    """Keep only achievement bullets written in that role's own part of the resume.

    Small models pad this field with generic lines ("Collaborated with teams to ensure quality")
    and copy bullets from neighbouring roles.
    """
    text_lower = text.lower()
    for item in profile.education_and_experience.experience:
        if not item.key_outcomes:
            continue
        anchor = item.company_name or item.designation
        window = _entry_window(
            text_lower, anchor, [n for n in all_names if n.lower() != anchor.lower()], _OUTCOME_WINDOW_CHARS
        ) if anchor else None
        if not window:
            item.key_outcomes = None
            continue
        squashed = _squash(window)
        kept = []
        for line in item.key_outcomes.splitlines():
            bullet = _BULLET_PREFIX_RE.sub("", line).strip()
            words = [w for w in _NON_ALNUM_RE.split(bullet.lower()) if len(w) >= 4]
            if words and sum(w in squashed for w in words) >= _OUTCOME_MIN_WORD_SHARE * len(words):
                kept.append(f"• {bullet}")
        item.key_outcomes = "\n".join(kept) or None


# ---------------------------------------------------------------------------
# Role text: description and key outcomes read from each role's part of the resume
# ---------------------------------------------------------------------------
# The model is not asked for these: it only ever copied the role's bullets, and those copies
# were ~40% of its output (on CPU, ~35 s per resume). The resume text has them verbatim.
_ROLE_MAX_LINES = 40
_DESCRIPTION_MAX_CHARS = 600
_MAX_OUTCOMES = 3
_HAS_BULLET_RE = re.compile(r"^\s*(?:[•\-*–·▪●◦➢►✓]+|o(?=\s))\s*")
_OUTCOME_VERB_RE = re.compile(
    r"\b(?:increas|reduc|improv|sav|achiev|deliver|optimi[sz]|grew|grow|boost|cut|accelerat|scal|automat|"
    r"launch|won|award|exceed|generat|lower|rais|migrat|led)\w*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|x\b|k\b|\+|cr|lakh|million|mn|users|clients|hours|days)|[₹$€£]\s*\d|\b\d{2,}\b")
_LABEL_ONLY_RE = re.compile(r"^(?:designation|role|position|title|duration|location)\s*[-–:]", re.IGNORECASE)


def _role_items(lines: list[str]) -> list[str]:
    """Bullets and sentences of a role, with lines that wrapped on the page joined back."""
    items: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        bullet = bool(_HAS_BULLET_RE.match(line))
        line = _HAS_BULLET_RE.sub("", line).strip()
        if not line:
            continue
        wrapped = items and not bullet and (line[0].islower() or not re.search(r"[.!?:]$", items[-1]))
        if wrapped:
            items[-1] = f"{items[-1]} {line}"
        else:
            items.append(line)
    return items


_RESULT_LABEL_RE = re.compile(r"^(?:key\s+)?(?:contributions?|achievements?|accomplishments?|impact|results?)\b", re.IGNORECASE)
_DUTY_LABEL_RE = re.compile(r"^(?:responsibilit\w*|project|client|reporting\s*to|team\s+size|environment|tech\w*)\b", re.IGNORECASE)


def _outcome_score(item: str) -> int:
    """Measured results first ("cut costs 30%"), then result verbs or a "Contribution –" label.
    Duties ("Responsibility –", "Project –") are description, not outcomes."""
    if _DUTY_LABEL_RE.match(item):
        return 0
    dated = _DATE_LIKE_RE.sub("", item)  # "2021" is a date, not a result
    return 2 * bool(_NUMBER_RE.search(dated)) + bool(_OUTCOME_VERB_RE.search(item) or _RESULT_LABEL_RE.match(item))


def _fill_role_text(profile: BONCProfileExtraction, text: str, all_names: list[str]) -> None:
    ee = profile.education_and_experience
    raw = text.splitlines()
    lines = [ln.lower() for ln in raw]
    names = [n.lower() for n in all_names]
    # Another role's title ends this role only where a line starts with it: titles like
    # "Software Developer" also appear inside bullets ("Designation – Consultant (Software Developer)").
    titles = [e.designation.lower() for e in ee.experience if e.designation and len(e.designation) >= 6]
    for exp in ee.experience:
        if not exp.company_name:
            continue
        key = exp.company_name.lower()
        start = next((i for i, ln in enumerate(lines) if key in ln), None)
        if start is None:
            continue
        own = {key} | ({exp.designation.lower()} if exp.designation else set())
        body: list[str] = []
        for i in range(start + 1, min(len(raw), start + 1 + _ROLE_MAX_LINES)):
            low, line = lines[i].strip(), raw[i].strip()
            bare = low.lstrip(" •-*–·")
            other = any(n in low for n in names if n not in own) or any(
                bare.startswith(t) for t in titles if t not in own
            )
            if other or (line and _ends_section(_LIST_BULLET_RE.sub("", line).strip()) and not _DATE_LIKE_RE.search(line)):
                break
            if exp.designation and low.strip(" •-*") == exp.designation.lower():
                continue  # the title line under the company
            body.append(raw[i])
        items = [it for it in _role_items(body) if not _LABEL_ONLY_RE.match(it) or len(it) > 60]
        items = [it for it in items if len(it) >= 15 and not (_DATE_LIKE_RE.search(it) and len(it) < 40)]
        if not items:
            continue
        ranked = sorted((i for i, it in enumerate(items) if _outcome_score(it)), key=lambda i: -_outcome_score(items[i]))
        chosen: list[int] = []
        used = 0
        for i in ranked:
            if len(chosen) == _MAX_OUTCOMES or used + len(items[i]) + 3 > KEY_OUTCOMES_MAX_CHARS:
                continue
            chosen.append(i)
            used += len(items[i]) + 3
        if chosen:
            exp.key_outcomes = "\n".join(f"• {items[i]}" for i in sorted(chosen))
        rest, size = [], 0
        for i, it in enumerate(items):
            if i in chosen:
                continue
            if size + len(it) > _DESCRIPTION_MAX_CHARS:
                break
            rest.append(it)
            size += len(it) + 1
        if rest:
            exp.description = "\n".join(rest)


# ---------------------------------------------------------------------------
# Education rows the model skipped
# ---------------------------------------------------------------------------
# The 3B model often lists only the degree and drops "10+2 from Rajasthan Board (2004)" rows;
# which rows it keeps changes with tiny numeric differences (CPU vs GPU, prompt cache). Rows in
# the resume's education section that start with a qualification are read here when no model
# entry covers them.
_EDU_HEAD_RE = re.compile(r"^(?:education(?:al)?(?:\s+(?:qualifications?|details|background))?|academics?(?:\s+\w+)?|qualifications?)\s*:?$", re.IGNORECASE)
_DEGREE_START_RE = re.compile(
    r"^(?:10th|12th|10\s*\+\s*2|ssc|hsc|sslc|hslc|x|xii|class\s+(?:x|xii|10|12)|matric\w*|intermediate|"
    r"(?:senior\s+|higher\s+)?secondary|diploma|b\.?\s?e\.?|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|b\.?\s?com|m\.?\s?com|"
    r"bca|mca|mba|bba|pgdm|ph\.?\s?d|bachelor\w*|master\w*|post\s*graduat\w*)(?=[\s.(,\-–]|$)",
    re.IGNORECASE,
)
_EDU_SPLIT_RE = re.compile(r"\s+(?:from|at)\s+|\s*[,|]\s*|\s+[-–—]\s+", re.IGNORECASE)
_GRADE_RE = re.compile(r"\d{1,2}(?:\.\d+)?\s*%(?:\s*\([^)]*\))?|\b(?:cgpa|gpa)\s*:?\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?", re.IGNORECASE)
_EDU_MAX_LINES = 15


def _fill_missing_education(profile: BONCProfileExtraction, text: str) -> None:
    ee = profile.education_and_experience
    lines = [_LIST_BULLET_RE.sub("", ln).strip() for ln in text.splitlines()]
    start = next((i for i, ln in enumerate(lines) if _EDU_HEAD_RE.match(ln)), None)
    if start is None:
        return
    def years(*values: str | None) -> set[str]:
        return {y for v in values if v for y in _YEAR_RE.findall(v)}

    # A row is the model's when its degree is in it ("B.E. (Computer)"), or its institute and
    # year both are: one board can award both the 10th and the 10+2 rows.
    covered = [
        (_squash(e.degree or ""), _squash(e.institute_name or ""), years(e.from_date, e.to_date)) for e in ee.education
    ]
    added = []
    for line in lines[start + 1: start + 1 + _EDU_MAX_LINES]:
        if not line:
            continue
        if _is_heading(line) or (line.upper() == line and len(line) <= 40 and not _DEGREE_START_RE.match(line)):
            break  # the next section
        m = _DEGREE_START_RE.match(line)
        if not m:
            continue
        key = _squash(line)
        year = _YEAR_RE.findall(line)
        if any(
            (len(deg) >= 2 and key.startswith(deg))  # "SSC with 87.80% May 2018" for the model's "SSC"
            or (len(deg) >= 4 and deg in key)
            or (len(inst) >= 5 and inst in key and (not yrs or yrs & set(year)))
            for deg, inst, yrs in covered
        ):
            continue  # the model has this row
        grade = _GRADE_RE.search(line)
        head = _GRADE_RE.sub("", re.sub(r"\(\s*(?:19|20)\d{2}\s*\)", "", line))
        parts = [p.strip(" .-–") for p in _EDU_SPLIT_RE.split(head) if p.strip(" .-–")]
        degree = parts[0] if parts else m.group(0)
        institute = next((p for p in parts[1:] if not _YEAR_RE.search(p) and len(p) >= 3), None)
        added.append({
            "degree": degree, "institute_name": institute, "to_date": year[-1] if year else None,
            "description": grade.group(0).strip() if grade else None,
        })
        covered.append((_squash(degree), _squash(institute or ""), set(year)))
    if added:
        ee.education = ee.education + [EducationItem.model_validate(a) for a in added]


# ---------------------------------------------------------------------------
# Profession, work-mode preferences, OCR-wrapped company names
# ---------------------------------------------------------------------------
_ROLE_WORD_RE = re.compile(
    r"\b(?:engineer|developer|manager|analyst|designer|consultant|specialist|administrator|scientist|"
    r"architect|tester|lead|officer|accountant|teacher|nurse|student|intern|owner|master|executive|"
    r"director|associate|technician|coordinator|programmer|writer|editor|marketer|recruiter|advocate|"
    r"lawyer|pharmacist|assistant|representative|qa|devops|professional|graduate|trainee|head)s?\b",
    re.IGNORECASE,
)


def _ground_profession(profile: BONCProfileExtraction, text: str) -> None:
    """The model often leaves ``profession`` empty. Take the title line under the name heading
    ("Ilze Berzkalne / Software Tester"), else the current or most recent job title."""
    networking = profile.networking
    if networking.profession:
        return
    first = (profile.basic_details.first_name or "").lower()
    lines = [ln.strip().lstrip("#*").strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines[:_NAME_SCAN_LINES]):
        if first and line.lower().startswith(first):
            title = lines[i + 1] if i + 1 < len(lines) else ""
            if 1 <= len(title.split()) <= 6 and not re.search(r"[\d@|:,/]", title) and _ROLE_WORD_RE.search(title):
                networking.profession = title
                return
            break
    roles = [e for e in profile.education_and_experience.experience if e.designation]
    current = next((e for e in roles if e.currently_working_here), roles[0] if roles else None)
    if current:
        networking.profession = current.designation


_WORK_MODE_NAMES = {"remote": "Remote", "hybrid": "Hybrid", "onsite": "On-Site", "inoffice": "On-Site"}
_WORK_MODE = r"(?:remote|hybrid|on[\s-]?site|in[\s-]?office)"
_WORK_MODE_LINE_RE = re.compile(
    rf"^[#*•\s]*(?:(?:preferred\s+)?work\s*(?:mode|preference)s?\s*[:\-]\s*)?"
    rf"({_WORK_MODE}(?:\s*(?:,|/|&|\band\b|\bor\b)\s*{_WORK_MODE})*)\s*"
    r"(?:[·|•,].*|\s+(?:permanent|temporary|full[\s-]?time|part[\s-]?time|contract|freelance|project[\s-]?based)\b.*)?$",
    re.IGNORECASE,
)
_PREFERENCE_HEADING_RE = re.compile(r"\b(?:preferences?|work\s*mode|looking\s+for)\b", re.IGNORECASE)


def _ground_work_modes(profile: BONCProfileExtraction, text: str) -> None:
    """Work modes the resume states as a preference ("Preferences / Remote, Hybrid · Part-time"
    or "Work mode: Remote"). A bare "Remote" elsewhere is usually a job's location, so it only
    counts under a preferences heading."""
    prefs = profile.career_preferences
    if prefs.work_modes:
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        # OCR of two-column resumes puts the section in table cells: "| Hybrid · Part-time |".
        cells = [c.strip() for c in line.strip("|").split("|")] if "|" in line else [line]
        m = next((m for c in cells if (m := _WORK_MODE_LINE_RE.match(c))), None)
        if not m:
            continue
        labelled = bool(re.match(r"^[#*•\s]*(?:preferred\s+)?work\s*(?:mode|preference)", line, re.IGNORECASE))
        if not labelled and not any(_PREFERENCE_HEADING_RE.search(prev) for prev in lines[max(0, i - 3): i]):
            continue
        modes = [_WORK_MODE_NAMES[re.sub(r"[\s-]", "", w.lower())] for w in re.findall(_WORK_MODE, m.group(1), re.IGNORECASE)]
        prefs.work_modes = list(dict.fromkeys(modes))
        prefs.work_mode = prefs.work_modes[0]
        return


_TYPE_AS_TITLE = {"fulltime", "parttime", "contract", "contractual", "internship", "freelance", "temporary", "permanent"}


def _ground_type_as_title(profile: BONCProfileExtraction) -> None:
    """"Full-time" written as the job title: it is the employment type; the title is unknown."""
    for exp in profile.education_and_experience.experience:
        if exp.designation and _squash(exp.designation) in _TYPE_AS_TITLE:
            exp.employment_type = exp.employment_type or _match_enum(exp.designation, EXP_EMPLOYMENT_OPTIONS)
            exp.designation = None
    ee = profile.education_and_experience
    ee.experience = [e for e in ee.experience if e.company_name or e.designation]


_SECTION_HEAD_RE = re.compile(
    r"^[#*\s]*(?:(?P<exp>(?:professional\s+|work\s+|relevant\s+)?(?:experience|employment(?:\s+history)?|work\s+history|internships?))"
    r"|(?P<proj>(?:academic\s+|personal\s+|key\s+)?projects?)"
    r"|(?P<other>education|skills|technical\s+skills|certifications?|summary|profile|achievements|awards|languages|hobbies|interests|references|publications))"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


def _sections(text: str) -> list[tuple[int, str]]:
    """(line index, kind) for each section heading line: "exp", "proj" or "other"."""
    out = []
    for i, line in enumerate(text.splitlines()):
        word = line.strip().lstrip("#* ").rstrip(": ")
        if word == "Project":
            continue  # a sub-label inside a job ("Sr. Developer, Acme / Project / ..."), not a section
        if len(line) <= 40 and (m := _SECTION_HEAD_RE.match(line.strip())):
            out.append((i, m.lastgroup))
    return out


def _drop_project_entries(profile: BONCProfileExtraction, text: str) -> None:
    """Projects listed as jobs: an entry whose name occurs only under a Projects heading (and
    the resume has an Experience section of its own) is a project, not an employer."""
    heads = _sections(text)
    if not any(k == "proj" for _, k in heads) or not any(k == "exp" for _, k in heads):
        return
    lines = [_squash(line) for line in text.splitlines()]

    def section_of(i: int) -> str | None:
        return next((k for j, k in reversed(heads) if j <= i), None)

    def only_in_projects(exp) -> bool:
        name = _squash(exp.company_name or "")
        return bool(name) and {section_of(i) for i, line in enumerate(lines) if name in line} == {"proj"}

    ee = profile.education_and_experience
    ee.experience = [e for e in ee.experience if not only_in_projects(e)]


def _closest_spelling(name: str, text: str) -> str | None:
    """The resume's spelling of a name the model copied with a typo ("Dowlabs" for "Devlabs")."""
    words = re.findall(r"[\w&.'-]+", text)
    n = len(name.split())
    if not n or not words:
        return None
    target = name.lower()
    best, best_ratio, ties = None, 0.0, 0
    for i in range(len(words) - n + 1):
        cand = " ".join(words[i: i + n])
        ratio = SequenceMatcher(None, target, cand.lower()).ratio()
        if ratio > best_ratio + 1e-9:
            best, best_ratio, ties = cand, ratio, 0
        elif abs(ratio - best_ratio) < 1e-9 and cand.lower() != (best or "").lower():
            ties += 1
    return best if best_ratio >= 0.85 and not ties else None


def _ground_spelling(profile: BONCProfileExtraction, text: str) -> None:
    squashed = _squash(text)
    ee = profile.education_and_experience
    for item, attr in [(e, "company_name") for e in ee.experience] + [(e, "institute_name") for e in ee.education]:
        name = getattr(item, attr)
        if name and _squash(name) not in squashed and (fixed := _closest_spelling(name, text)):
            setattr(item, attr, fixed)


def _ground_wrapped_companies(profile: BONCProfileExtraction, text: str) -> None:
    """OCR of two-column resumes yields tables whose cells wrap: "| Senior DBA — Baltic | Dec 2023 |"
    then "| Vault Data Services | 2026 |". The model keeps only the second half of the company."""
    if "|" not in text:
        return
    for exp in profile.education_and_experience.experience:
        company = exp.company_name
        if not company:
            continue
        m = re.search(
            r"(?:—|–|\\|\s-\s|\bat\s)\s*([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3})\s*\|[^\n]*\n"
            r"(?:\|[-:|\s]*\|\s*\n)?\|\s*" + re.escape(company) + r"\s*\|",
            text,
        )
        if m and _squash(m.group(1)) not in _squash(company):
            exp.company_name = f"{m.group(1)} {company}"


# ---------------------------------------------------------------------------
# Address from the contact line
# ---------------------------------------------------------------------------
_INDIAN_STATES = {
    "andhrapradesh": "Andhra Pradesh", "ap": "Andhra Pradesh", "arunachalpradesh": "Arunachal Pradesh",
    "assam": "Assam", "bihar": "Bihar", "chhattisgarh": "Chhattisgarh", "cg": "Chhattisgarh", "goa": "Goa",
    "gujarat": "Gujarat", "gj": "Gujarat", "haryana": "Haryana", "hr": "Haryana",
    "himachalpradesh": "Himachal Pradesh", "hp": "Himachal Pradesh", "jharkhand": "Jharkhand",
    "karnataka": "Karnataka", "ka": "Karnataka", "kerala": "Kerala", "kl": "Kerala",
    "madhyapradesh": "Madhya Pradesh", "mp": "Madhya Pradesh", "maharashtra": "Maharashtra", "mh": "Maharashtra",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram", "nagaland": "Nagaland",
    "odisha": "Odisha", "orissa": "Odisha", "punjab": "Punjab", "pb": "Punjab", "rajasthan": "Rajasthan",
    "rj": "Rajasthan", "sikkim": "Sikkim", "tamilnadu": "Tamil Nadu", "tn": "Tamil Nadu",
    "telangana": "Telangana", "ts": "Telangana", "tripura": "Tripura", "uttarpradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh", "uttarakhand": "Uttarakhand", "uk": "Uttarakhand", "westbengal": "West Bengal",
    "wb": "West Bengal", "delhi": "Delhi", "newdelhi": "Delhi", "dl": "Delhi", "ncr": "Delhi",
    "jammuandkashmir": "Jammu and Kashmir", "jk": "Jammu and Kashmir", "ladakh": "Ladakh",
    "chandigarh": "Chandigarh", "puducherry": "Puducherry", "pondicherry": "Puducherry",
}
_COUNTRIES = {"india": "India", "bharat": "India", "usa": "United States", "unitedstates": "United States",
              "uk": "United Kingdom", "unitedkingdom": "United Kingdom", "uae": "United Arab Emirates",
              "canada": "Canada", "australia": "Australia", "germany": "Germany", "singapore": "Singapore"}
_PLACE_PART = r"[A-Z][A-Za-z .'-]{1,30}"
_CONTACT_PLACE_RE = re.compile(
    rf"^(?:(?:address|location|city)\s*[:\-]\s*)?({_PLACE_PART}),\s*({_PLACE_PART})(?:,\s*({_PLACE_PART}))?(?:[,\s-]+(\d{{6}}))?$",
    re.IGNORECASE,
)


def _contact_place(text: str) -> dict[str, str] | None:
    """City/state/country written in the header ("Coimbatore, Tamil Nadu", "Aurangabad, MH, India")."""
    lines = [ln.strip().lstrip("#*").strip() for ln in text.splitlines() if ln.strip()][:_NAME_SCAN_LINES]
    for line in lines:
        # Icons often come out as ":" (" : Coimbatore, Tamil Nadu :+91 ..."), so " :" separates too.
        for seg in re.split(r"\s*(?:\||·|•|—|–|\s-\s|\s:)\s*|\s{2,}", line):
            m = _CONTACT_PLACE_RE.match(seg.strip(" :"))
            if not m:
                continue
            city, second, third, pin = (m.group(i) for i in range(1, 5))
            key2, key3 = _squash(second), _squash(third or "")
            if key2 in _INDIAN_STATES and (not third or _COUNTRIES.get(key3) == "India"):
                place = {"city": city.strip(), "state": _INDIAN_STATES[key2], "country": "India"}
            elif key2 in _COUNTRIES and not third and key2 != "uk":
                place = {"city": city.strip(), "country": _COUNTRIES[key2]}
            else:
                continue
            if pin:
                place["pincode"] = pin
            return place
    return None


def _ground_address(profile: BONCProfileExtraction, text: str) -> None:
    """Fill the primary address from the header's "City, State[, Country]" when the model left
    the city empty or put it in the street field."""
    place = _contact_place(text)
    if not place:
        return
    addr = profile.basic_details.primary_address
    city = _squash(place["city"])
    if addr.city and _squash(addr.city) != city:
        return  # the model read a different (likely fuller) address: keep it
    if addr.street_area_locality and _squash(addr.street_area_locality) == city:
        addr.street_area_locality = None
    for field, value in place.items():
        current = getattr(addr, field)
        # A state abbreviation ("MH") becomes its full name.
        if not current or (field == "state" and _INDIAN_STATES.get(_squash(current)) == value):
            setattr(addr, field, value)


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------
_PHONE_MIN_DIGITS = 7
_PHONE_NATIONAL_DIGITS = 10


def _phone_key(number: str | None) -> str | None:
    """Digits that identify a phone number regardless of formatting and country prefix:
    "+91 98765 43210", "0091-9876543210", "09876543210" and "9876543210" share one key."""
    digits = re.sub(r"\D", "", number or "")
    if len(digits) < _PHONE_MIN_DIGITS:
        return None
    return digits[-_PHONE_NATIONAL_DIGITS:]


def same_phone_number(a: str | None, b: str | None) -> bool:
    key = _phone_key(a)
    return key is not None and key == _phone_key(b)


def _drop_primary_phone(profile: BONCProfileExtraction, primary_phone: str | None) -> None:
    """The resume's number is usually the one the user signed up with; it is not an alternate."""
    basic = profile.basic_details
    if same_phone_number(basic.alternate_phone_number, primary_phone):
        basic.alternate_phone_number = None


def ground_profile(
    profile: BONCProfileExtraction, resume_text: str, primary_phone: str | None = None
) -> BONCProfileExtraction:
    """Correct names, links and ongoing flags in place using evidence from ``resume_text``, and
    clear the alternate phone number when it is the user's primary (registered) number."""
    _drop_primary_phone(profile, primary_phone)
    if not resume_text:
        return profile
    _ground_name(profile, resume_text)
    _ground_links(profile, resume_text)
    _ground_lists(profile, resume_text)
    _fill_missing_education(profile, resume_text)
    _ground_type_as_title(profile)
    _ground_spelling(profile, resume_text)
    _ground_wrapped_companies(profile, resume_text)
    _drop_project_entries(profile, resume_text)
    ee = profile.education_and_experience
    all_names = [e.company_name for e in ee.experience if e.company_name]
    all_names += [e.institute_name for e in ee.education if e.institute_name]
    _ground_ongoing(ee.experience, "company_name", "currently_working_here", resume_text, all_names)
    _ground_ongoing(ee.education, "institute_name", "currently_studying_here", resume_text, all_names)
    _ground_entry_details(profile, resume_text, all_names)
    _ground_outcomes(profile, resume_text, all_names)
    _fill_role_text(profile, resume_text, all_names)
    _ground_profession(profile, resume_text)
    _ground_work_modes(profile, resume_text)
    _ground_address(profile, resume_text)
    # Autofill stops at 6 secondary skills (after the section top-up and the tools move); the
    # user adds any others by hand.
    ee.skills.secondary = ee.skills.secondary[:AUTOFILL_MAX_SECONDARY_SKILLS]
    return profile
