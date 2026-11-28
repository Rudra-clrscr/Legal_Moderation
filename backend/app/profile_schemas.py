"""Pydantic schemas for BONC profile extraction from resumes."""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations (must match BONCProfileExtraction in profile.html)
# ---------------------------------------------------------------------------
GENDER_OPTIONS = ("Male", "Female", "Other")
SOCIAL_PLATFORM_OPTIONS = ("LinkedIn", "Twitter", "Instagram", "Facebook", "GitHub", "Website", "Other")
EXP_EMPLOYMENT_OPTIONS = ("Full-time", "Part-time", "Contract", "Internship")
CAREER_EMPLOYMENT_OPTIONS = ("Full-Time", "Part-Time", "Contract", "Internship", "Freelance")
WORK_MODE_OPTIONS = ("Hybrid", "Remote", "On-Site")
CAREER_MODE_OPTIONS = ("Looking for Job", "Looking for Talent")
JOB_CATEGORY_OPTIONS = (
    "Information Technology",
    "Sales & Marketing",
    "Finance & Accounting",
    "Human Resources",
    "Operations & Logistics",
    "Manufacturing & Production",
    "Engineering",
    "Healthcare & Medical",
    "Education & Training",
    "Design & Creative",
    "Legal & Compliance",
    "Customer Support",
    "Business Development",
    "Research & Development",
    "Supply Chain",
)

Gender = Literal["Male", "Female", "Other"]
SocialPlatform = Literal["LinkedIn", "Twitter", "Instagram", "Facebook", "GitHub", "Website", "Other"]
ExperienceEmploymentType = Literal["Full-time", "Part-time", "Contract", "Internship"]
CareerEmploymentType = Literal["Full-Time", "Part-Time", "Contract", "Internship", "Freelance"]
WorkMode = Literal["Hybrid", "Remote", "On-Site"]
CareerMode = Literal["Looking for Job", "Looking for Talent"]
JobCategory = Literal[
    "Information Technology",
    "Sales & Marketing",
    "Finance & Accounting",
    "Human Resources",
    "Operations & Logistics",
    "Manufacturing & Production",
    "Engineering",
    "Healthcare & Medical",
    "Education & Training",
    "Design & Creative",
    "Legal & Compliance",
    "Customer Support",
    "Business Development",
    "Research & Development",
    "Supply Chain",
]

# Tier-1 industry clusters: a fixed list so recruiter search can index on it.
INDUSTRY_CLUSTER_OPTIONS = (
    "Technology & Software",
    "Banking, Financial Services & Insurance",
    "Healthcare & Pharma",
    "Consumer & Retail",
    "Logistics & Supply Chain",
    "Manufacturing & Industrial",
    "Energy & Utilities",
    "Telecom & Media",
    "Education & EdTech",
    "Real Estate & Construction",
    "Automotive & Mobility",
    "Travel, Hospitality & Food Services",
    "Professional Services & Consulting",
    "Government & Public Sector",
    "Agriculture & Food Processing",
    "Non-Profit & Social Impact",
)
LANGUAGE_PROFICIENCY_OPTIONS = ("Elementary", "Conversational", "Professional", "Fluent", "Native / Bilingual")
SKILL_PROFICIENCY_MIN, SKILL_PROFICIENCY_MAX = 1, 10
MAX_PRIMARY_SKILLS = 5
MAX_SECONDARY_SKILLS = 15
# Resume autofill fills at most this many secondary skills; the user can add the rest (up to
# MAX_SECONDARY_SKILLS) by hand. Product decision, 2026-11-27.
AUTOFILL_MAX_SECONDARY_SKILLS = 6
KEY_OUTCOMES_MAX_CHARS = 500

IndustryCluster = Literal[INDUSTRY_CLUSTER_OPTIONS]  # type: ignore[valid-type]
LanguageProficiency = Literal[LANGUAGE_PROFICIENCY_OPTIONS]  # type: ignore[valid-type]
_LANGUAGE_SYNONYMS = {
    "basic": "Elementary", "beginner": "Elementary", "elementaryproficiency": "Elementary",
    "intermediate": "Conversational", "limitedworkingproficiency": "Conversational",
    "professionalworkingproficiency": "Professional", "working": "Professional",
    "fullprofessionalproficiency": "Fluent", "advanced": "Fluent", "proficient": "Fluent", "expert": "Fluent",
    "native": "Native / Bilingual", "bilingual": "Native / Bilingual", "mothertongue": "Native / Bilingual",
    "nativeorbilingualproficiency": "Native / Bilingual",
}

_ENUM_KEY_RE = re.compile(r"[^a-z0-9]")
_ONGOING_TOKENS = {"present", "current", "currently", "ongoing", "now", "till date", "to date"}
_ADDRESS_FIELDS = (
    "house_flat_building",
    "street_area_locality",
    "landmark",
    "country",
    "state",
    "city",
    "pincode",
)


# Filler the model writes instead of leaving a field null.
_PLACEHOLDERS = {"unknown", "n/a", "none", "null", "nil", "not specified", "not mentioned", "not available", "not provided"}


def _clean_str(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text if text and text.lower() not in _PLACEHOLDERS else None
    return None


# OCR reads the digit 0 as the letter o ("+1 555-o114").
_PHONE_CHARS_RE = re.compile(r"[+\d\s().-]+")


def _clean_phone(value: Any) -> str | None:
    text = _clean_str(value)
    if text and not _PHONE_CHARS_RE.fullmatch(text):
        fixed = re.sub(r"(?<=[\d\s-])[oO]|[oO](?=\d)", "0", text)
        if _PHONE_CHARS_RE.fullmatch(fixed):
            return fixed
    return text


# OCR letter-spaces years in some layouts: "2 0 1 4".
_SPACED_YEAR_RE = re.compile(r"\b([12]) (\d) (\d) (\d)\b")


def _match_enum(value: Any, options: tuple[str, ...]) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None
    key = _ENUM_KEY_RE.sub("", text.lower())
    if not key:
        return None
    for option in options:
        if _ENUM_KEY_RE.sub("", option.lower()) == key:
            return option
    return None


def _clean_tags(value: Any, limit: int, max_len: int = 80) -> list[str]:
    """A de-duplicated (case-insensitive) list of short strings; accepts a list or a comma list."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_str(item)
        if not text or len(text) > max_len:
            continue
        key = _ENUM_KEY_RE.sub("", text.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out[:limit]


def _clean_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num < 0:
        return None
    return num


def _clean_int(value: Any) -> int | None:
    num = _clean_float(value)
    return round(num) if num is not None else None


# "Oct 2013 - May 2016", "2005 – 2009", "Oct 2022-May 2024", "Jan 2021 to Present" in one field.
_RANGE_SPLIT_RE = re.compile(r"\s+(?:-|to|till|until)\s+|\s*[–—]\s*|(?<=\d{4})-(?=[A-Za-z])|(?<=[A-Za-z]\s\d{4})-(?=\d)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _split_range(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parts = _RANGE_SPLIT_RE.split(value, maxsplit=1)
    if len(parts) != 2:
        return None
    start, end = parts[0].strip(), parts[1].strip()
    if _YEAR_RE.search(start) and (_YEAR_RE.search(end) or end.lower() in _ONGOING_TOKENS):
        return start, end
    return None


def _normalize_period(data: Any, current_flag: str) -> Any:
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key in ("from_date", "to_date"):
        if isinstance(out.get(key), str):
            out[key] = _SPACED_YEAR_RE.sub(r"\1\2\3\4", out[key])
    # Small models put a whole date range into one field; split it into from/to.
    from_date, to_date = _clean_str(out.get("from_date")), _clean_str(out.get("to_date"))
    if (rng := _split_range(from_date)) and (not to_date or to_date == from_date):
        out["from_date"], out["to_date"] = rng
    elif (rng := _split_range(to_date)) and (not from_date or from_date in (rng[0], to_date)):
        out["from_date"], out["to_date"] = rng
    to_date = _clean_str(out.get("to_date"))
    if to_date and to_date.lower() in _ONGOING_TOKENS:
        out[current_flag] = True
        out["to_date"] = None
    if out.get(current_flag) is True:
        out["to_date"] = None
    return out


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------
class Address(BaseModel):
    house_flat_building: str | None = None
    street_area_locality: str | None = None
    landmark: str | None = None
    country: str | None = None
    state: str | None = None
    city: str | None = None
    pincode: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _clean_fields(cls, value: Any, info) -> Any:
        if info.field_name in _ADDRESS_FIELDS:
            return _clean_str(value)
        return value


class AdditionalAddress(Address):
    set_as_default_address: bool = False

    @field_validator("set_as_default_address", mode="before")
    @classmethod
    def _coerce_default_flag(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False


class BasicDetails(BaseModel):
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
    alternate_phone_number: str | None = None
    gender: Gender | None = None
    primary_address: Address = Field(default_factory=Address)
    additional_addresses: list[AdditionalAddress] = Field(default_factory=list)

    @field_validator("first_name", "middle_name", "last_name", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("alternate_phone_number", mode="before")
    @classmethod
    def _clean_phone_number(cls, value: Any) -> str | None:
        return _clean_phone(value)

    @field_validator("gender", mode="before")
    @classmethod
    def _clean_gender(cls, value: Any) -> str | None:
        return _match_enum(value, GENDER_OPTIONS)

    @field_validator("primary_address", mode="before")
    @classmethod
    def _coerce_primary_address(cls, value: Any) -> Any:
        return value if isinstance(value, (dict, Address)) else {}

    @field_validator("additional_addresses", mode="before")
    @classmethod
    def _coerce_additional_addresses(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, (dict, AdditionalAddress))]

    @model_validator(mode="after")
    def _filter_additional_addresses(self) -> "BasicDetails":
        non_empty = [
            addr
            for addr in self.additional_addresses
            if any(getattr(addr, field) is not None for field in _ADDRESS_FIELDS)
        ]
        self.additional_addresses = non_empty[:10]
        return self


_URL_PLATFORMS = (
    ("linkedin.com", "LinkedIn"),
    ("github.com", "GitHub"),
    ("instagram.com", "Instagram"),
    ("facebook.com", "Facebook"),
    ("fb.com", "Facebook"),
    ("twitter.com", "Twitter"),
    ("x.com", "Twitter"),
)


def platform_from_url(url: str) -> str | None:
    """Platform implied by a well-known domain, or None for other sites."""
    host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url.strip().lower()).split("/")[0].split(":")[0]
    for domain, platform in _URL_PLATFORMS:
        if host == domain or host.endswith("." + domain):
            return platform
    return None


class SocialLink(BaseModel):
    platform: SocialPlatform = "Other"
    url: str

    @model_validator(mode="after")
    def _trust_url_over_label(self) -> "SocialLink":
        # Small models often copy the prompt's example label ("LinkedIn") onto any link.
        self.platform = platform_from_url(self.url) or self.platform
        return self

    @field_validator("platform", mode="before")
    @classmethod
    def _clean_platform(cls, value: Any) -> str:
        return _match_enum(value, SOCIAL_PLATFORM_OPTIONS) or "Other"

    @field_validator("url", mode="before")
    @classmethod
    def _clean_url(cls, value: Any) -> str:
        cleaned = _clean_str(value)
        if not cleaned:
            raise ValueError("Social link URL cannot be empty.")
        return cleaned


class Networking(BaseModel):
    profession: str | None = None
    location: str | None = None
    date_of_birth: str | None = None
    tag_line: str | None = None
    about: str | None = None
    social_links: list[SocialLink] = Field(default_factory=list)

    @field_validator("profession", "location", "date_of_birth", "tag_line", "about", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("social_links", mode="before")
    @classmethod
    def _coerce_social_links(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        valid: list[Any] = []
        for item in value:
            if isinstance(item, SocialLink):
                valid.append(item)
            elif isinstance(item, dict) and _clean_str(item.get("url")):
                valid.append(item)
        return valid


class EducationItem(BaseModel):
    institute_name: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    location: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    currently_studying_here: bool = False
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        return _normalize_period(data, "currently_studying_here")

    @field_validator(
        "institute_name",
        "degree",
        "field_of_study",
        "location",
        "from_date",
        "to_date",
        "description",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("currently_studying_here", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False


class ExperienceItem(BaseModel):
    company_name: str | None = None
    designation: str | None = None
    employment_type: ExperienceEmploymentType | None = None
    location: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    currently_working_here: bool = False
    description: str | None = None
    key_outcomes: str | None = None  # measurable achievements, "• " bullets, up to 500 chars

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        return _normalize_period(data, "currently_working_here")

    @field_validator(
        "company_name",
        "designation",
        "location",
        "from_date",
        "to_date",
        "description",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("key_outcomes", mode="before")
    @classmethod
    def _clean_outcomes(cls, value: Any) -> str | None:
        if isinstance(value, list):  # models often return bullets as a list
            value = "\n".join(f"• {str(v).strip().lstrip('•-* ').strip()}" for v in value if _clean_str(v))
        text = _clean_str(value)
        return text[:KEY_OUTCOMES_MAX_CHARS].rstrip() if text else None

    @field_validator("employment_type", mode="before")
    @classmethod
    def _clean_employment_type(cls, value: Any) -> str | None:
        return _match_enum(value, EXP_EMPLOYMENT_OPTIONS)

    @field_validator("currently_working_here", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False


class SkillItem(BaseModel):
    name: str
    proficiency: int | None = None  # 1-10: 1-3 Basic, 4-6 Intermediate, 7-8 Advanced, 9-10 Master
    years: float | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        text = _clean_str(value)
        if not text or len(text) > 80:
            raise ValueError("Skill name must be 1-80 characters.")
        return text

    @field_validator("proficiency", mode="before")
    @classmethod
    def _clean_proficiency(cls, value: Any) -> int | None:
        num = _clean_int(value)
        return num if num is not None and SKILL_PROFICIENCY_MIN <= num <= SKILL_PROFICIENCY_MAX else None

    @field_validator("years", mode="before")
    @classmethod
    def _clean_years(cls, value: Any) -> float | None:
        num = _clean_float(value)
        return round(num, 1) if num is not None and num <= 60 else None


def _coerce_skill_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            item = {"name": item}
        if isinstance(item, (dict, SkillItem)):
            try:
                out.append(SkillItem.model_validate(item))
            except ValueError:
                continue
    return out


class Skills(BaseModel):
    primary: list[SkillItem] = Field(default_factory=list)
    secondary: list[SkillItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_list(cls, data: Any) -> Any:
        # The model returns one ranked list; the first five become primary skills.
        if isinstance(data, (list, str)):
            return {"primary": _coerce_skill_list(data), "secondary": []}  # split after de-duplication
        return data if isinstance(data, dict) else {}

    @field_validator("primary", "secondary", mode="before")
    @classmethod
    def _coerce_items(cls, value: Any) -> list[Any]:
        return _coerce_skill_list(value)

    @model_validator(mode="after")
    def _dedupe_and_cap(self) -> "Skills":
        seen: set[str] = set()

        def keep(items: list[SkillItem]) -> list[SkillItem]:
            out = []
            for s in items:
                key = _ENUM_KEY_RE.sub("", s.name.lower())
                if key and key not in seen:
                    seen.add(key)
                    out.append(s)
            return out

        primary = keep(self.primary)
        secondary = keep(self.secondary)
        # Overflow from primary moves down rather than being lost.
        secondary = primary[MAX_PRIMARY_SKILLS:] + secondary
        self.primary = primary[:MAX_PRIMARY_SKILLS]
        self.secondary = secondary[:MAX_SECONDARY_SKILLS]
        return self


class LanguageItem(BaseModel):
    language: str
    proficiency: LanguageProficiency | None = None

    @field_validator("language", mode="before")
    @classmethod
    def _clean_language(cls, value: Any) -> str:
        text = _clean_str(value)
        if not text or len(text) > 40:
            raise ValueError("Language must be 1-40 characters.")
        return text

    @field_validator("proficiency", mode="before")
    @classmethod
    def _clean_proficiency(cls, value: Any) -> str | None:
        matched = _match_enum(value, LANGUAGE_PROFICIENCY_OPTIONS)
        if matched:
            return matched
        key = _ENUM_KEY_RE.sub("", (_clean_str(value) or "").lower())
        return _LANGUAGE_SYNONYMS.get(key)


class CertificationItem(BaseModel):
    name: str
    issuer: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    does_not_expire: bool = False
    credential_url: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        text = _clean_str(value)
        if not text:
            raise ValueError("Certification name is required.")
        return text

    @field_validator("issuer", "issue_date", "expiry_date", "credential_url", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("does_not_expire", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False

    @model_validator(mode="after")
    def _no_expiry_clears_date(self) -> "CertificationItem":
        if self.expiry_date and self.expiry_date.lower() in {"none", "never", "no expiry", "does not expire", "lifetime"}:
            self.expiry_date, self.does_not_expire = None, True
        if self.does_not_expire:
            self.expiry_date = None
        return self


def _coerce_model_list(value: Any, model: type[BaseModel], limit: int, scalar_key: str | None = None) -> list[Any]:
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        if scalar_key and isinstance(item, str):
            item = {scalar_key: item}
        if isinstance(item, (dict, model)):
            try:
                out.append(model.model_validate(item))
            except ValueError:
                continue
    return out[:limit]


_LIST_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "skills": ("technical_skills", "key_skills", "core_skills", "skill_set", "skillset", "core_competencies"),
    "tools": ("tools_and_technologies", "technologies", "frameworks", "tools_and_frameworks"),
    "languages": ("languages_known", "language_known", "spoken_languages"),
    "certifications": ("certificates", "certification", "licenses_and_certifications"),
}


class EducationAndExperience(BaseModel):
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    tools: list[str] = Field(default_factory=list)  # tools, frameworks & systems
    industry_clusters: list[IndustryCluster] = Field(default_factory=list)  # Tier 1, fixed list
    sub_domains: list[str] = Field(default_factory=list)  # Tier 2, free text
    languages: list[LanguageItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_alias_keys(cls, data: Any) -> Any:
        # Small models rename keys ("languages_known", "technical_skills"); keep their values.
        if isinstance(data, dict):
            data = dict(data)
            for key, aliases in _LIST_KEY_ALIASES.items():
                if not data.get(key):
                    found = next((data[a] for a in aliases if data.get(a)), None)
                    if found is not None:
                        data[key] = found
        return data

    @field_validator("skills", mode="before")
    @classmethod
    def _coerce_skills(cls, value: Any) -> Any:
        return value if isinstance(value, (dict, list, str, Skills)) else {}

    @field_validator("tools", mode="before")
    @classmethod
    def _clean_tools(cls, value: Any) -> list[str]:
        return _clean_tags(value, 30)

    @field_validator("sub_domains", mode="before")
    @classmethod
    def _clean_sub_domains(cls, value: Any) -> list[str]:
        return _clean_tags(value, 10)

    @field_validator("industry_clusters", mode="before")
    @classmethod
    def _clean_clusters(cls, value: Any) -> list[str]:
        matched = [_match_enum(v, INDUSTRY_CLUSTER_OPTIONS) for v in _clean_tags(value, 20)]
        return list(dict.fromkeys(m for m in matched if m))[:3]

    @field_validator("languages", mode="before")
    @classmethod
    def _coerce_languages(cls, value: Any) -> list[Any]:
        if isinstance(value, dict):  # {"English": "Fluent", "Hindi": "Native"}
            value = [{"language": k, "proficiency": v} for k, v in value.items()]
        items = _coerce_model_list(value, LanguageItem, 10, scalar_key="language")
        seen: set[str] = set()
        return [i for i in items if not (i.language.lower() in seen or seen.add(i.language.lower()))]

    @field_validator("certifications", mode="before")
    @classmethod
    def _coerce_certifications(cls, value: Any) -> list[Any]:
        return _coerce_model_list(value, CertificationItem, 20, scalar_key="name")

    @field_validator("education", mode="before")
    @classmethod
    def _coerce_education(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, (dict, EducationItem))]

    @field_validator("experience", mode="before")
    @classmethod
    def _coerce_experience(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, (dict, ExperienceItem))]

    @model_validator(mode="after")
    def _filter_empty_entries(self) -> "EducationAndExperience":
        self.education = _unique(
            [e for e in self.education if e.institute_name or e.degree],
            lambda e: (e.institute_name, e.degree, e.from_date, e.to_date),
        )
        # A repetition loop writes the same role again and again.
        self.experience = _unique(
            [e for e in self.experience if e.company_name or e.designation],
            lambda e: (e.company_name, e.designation, e.from_date, e.to_date),
        )
        return self


def _unique(items: list, key) -> list:
    seen: set = set()
    out = []
    for item in items:
        k = tuple(_ENUM_KEY_RE.sub("", (v or "").lower()) for v in key(item))
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


class CareerPreferences(BaseModel):
    mode: CareerMode | None = None
    job_category: JobCategory | None = None
    employment_type: CareerEmploymentType | None = None
    work_mode: WorkMode | None = None
    preferred_country: str | None = None
    preferred_state: str | None = None
    preferred_city: str | None = None
    experience_years_min: float | None = None
    experience_years_max: float | None = None
    salary_inr_per_year_min: int | None = None
    salary_inr_per_year_max: int | None = None
    target_roles: list[str] = Field(default_factory=list)
    work_modes: list[WorkMode] = Field(default_factory=list)  # multi-select; work_mode is the first
    preferred_locations: list[str] = Field(default_factory=list)
    open_to_relocation: bool = False

    @field_validator("target_roles", mode="before")
    @classmethod
    def _clean_roles(cls, value: Any) -> list[str]:
        return _clean_tags(value, 10, max_len=100)

    @field_validator("preferred_locations", mode="before")
    @classmethod
    def _clean_locations(cls, value: Any) -> list[str]:
        return _clean_tags(value, 10)

    @field_validator("work_modes", mode="before")
    @classmethod
    def _clean_work_modes(cls, value: Any) -> list[str]:
        matched = [_match_enum(v, WORK_MODE_OPTIONS) for v in _clean_tags(value, 10)]
        return list(dict.fromkeys(m for m in matched if m))

    @field_validator("open_to_relocation", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False

    @model_validator(mode="after")
    def _sync_work_mode(self) -> "CareerPreferences":
        # Keep the original single work_mode field meaningful for existing consumers.
        if self.work_mode and self.work_mode not in self.work_modes:
            self.work_modes.insert(0, self.work_mode)
        self.work_mode = self.work_modes[0] if self.work_modes else None
        return self

    @field_validator("mode", mode="before")
    @classmethod
    def _clean_mode(cls, value: Any) -> str | None:
        return _match_enum(value, CAREER_MODE_OPTIONS)

    @field_validator("job_category", mode="before")
    @classmethod
    def _clean_job_category(cls, value: Any) -> str | None:
        return _match_enum(value, JOB_CATEGORY_OPTIONS)

    @field_validator("employment_type", mode="before")
    @classmethod
    def _clean_employment_type(cls, value: Any) -> str | None:
        return _match_enum(value, CAREER_EMPLOYMENT_OPTIONS)

    @field_validator("work_mode", mode="before")
    @classmethod
    def _clean_work_mode(cls, value: Any) -> str | None:
        return _match_enum(value, WORK_MODE_OPTIONS)

    @field_validator("preferred_country", "preferred_state", "preferred_city", mode="before")
    @classmethod
    def _clean_location(cls, value: Any) -> str | None:
        return _clean_str(value)

    @field_validator("experience_years_min", "experience_years_max", mode="before")
    @classmethod
    def _clean_years(cls, value: Any) -> float | None:
        return _clean_float(value)

    @field_validator("salary_inr_per_year_min", "salary_inr_per_year_max", mode="before")
    @classmethod
    def _clean_salary(cls, value: Any) -> int | None:
        return _clean_int(value)


class BONCProfileExtraction(BaseModel):
    basic_details: BasicDetails = Field(default_factory=BasicDetails)
    networking: Networking = Field(default_factory=Networking)
    education_and_experience: EducationAndExperience = Field(default_factory=EducationAndExperience)
    career_preferences: CareerPreferences = Field(default_factory=CareerPreferences)
    profile_image: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_misplaced_lists(cls, data: Any) -> Any:
        # Skills or languages written at the top level or in an invented section
        # ("personal_details") belong in education_and_experience.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        ee = data.get("education_and_experience")
        ee = dict(ee) if isinstance(ee, dict) else {}
        sources = [data] + [v for k, v in data.items() if isinstance(v, dict) and k != "education_and_experience"]
        for key, aliases in _LIST_KEY_ALIASES.items():
            if any(ee.get(k) for k in (key, *aliases)):
                continue
            for src in sources:
                found = next((src[k] for k in (key, *aliases) if src.get(k)), None)
                if found is not None:
                    ee[key] = found
                    break
        data["education_and_experience"] = ee
        return data

    @field_validator(
        "basic_details",
        "networking",
        "education_and_experience",
        "career_preferences",
        mode="before",
    )
    @classmethod
    def _coerce_section(cls, value: Any) -> Any:
        if isinstance(value, (dict, BaseModel)):
            return value
        return {}

    @field_validator("profile_image", mode="before")
    @classmethod
    def _clean_profile_image(cls, value: Any) -> str | None:
        cleaned = _clean_str(value)
        if cleaned and cleaned.startswith("data:image/"):
            return cleaned
        return None
