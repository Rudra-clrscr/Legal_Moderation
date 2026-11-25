"""Deterministic corrections to LLM profile output, checked against the resume text itself.

A small local model reliably gets the gist of a resume but makes a few repeatable mistakes:
splitting a three-part name as first + last, copying the prompt's placeholder URL, skipping
links, and leaving ``currently_working_here`` false on "Jan 2021 - Present" roles. Each of these
facts is plainly visible in the text, so we fix them here without guessing: every correction
needs direct evidence in ``resume_text``.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.profile_schemas import BONCProfileExtraction, SocialLink, platform_from_url

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
    if b.middle_name:
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
    links: list[SocialLink] = []
    seen: set[str] = set()

    model_keys = {_link_key(link.url) for link in profile.networking.social_links}
    for link in profile.networking.social_links:
        url = link.url
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


def _ground_certifications(profile: BONCProfileExtraction, text: str) -> None:
    for cert in profile.education_and_experience.certifications:
        # "PMP - Project Management Institute": the issuer belongs in its own field.
        if cert.issuer:
            for sep in (" - ", " – ", " — ", ", "):
                suffix = f"{sep}{cert.issuer}"
                if cert.name.lower().endswith(suffix.lower()) and len(cert.name) > len(suffix):
                    cert.name = cert.name[: -len(suffix)].strip()
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


def ground_profile(profile: BONCProfileExtraction, resume_text: str) -> BONCProfileExtraction:
    """Correct names, links and ongoing flags in place using evidence from ``resume_text``."""
    if not resume_text:
        return profile
    _ground_name(profile, resume_text)
    _ground_links(profile, resume_text)
    _ground_lists(profile, resume_text)
    ee = profile.education_and_experience
    all_names = [e.company_name for e in ee.experience if e.company_name]
    all_names += [e.institute_name for e in ee.education if e.institute_name]
    _ground_ongoing(ee.experience, "company_name", "currently_working_here", resume_text, all_names)
    _ground_ongoing(ee.education, "institute_name", "currently_studying_here", resume_text, all_names)
    _ground_outcomes(profile, resume_text, all_names)
    return profile
