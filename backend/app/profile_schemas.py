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


def _clean_str(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


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


def _normalize_period(data: Any, current_flag: str) -> Any:
    if not isinstance(data, dict):
        return data
    out = dict(data)
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

    @field_validator("first_name", "middle_name", "last_name", "alternate_phone_number", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        return _clean_str(value)

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


class SocialLink(BaseModel):
    platform: SocialPlatform = "Other"
    url: str

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

    @field_validator("employment_type", mode="before")
    @classmethod
    def _clean_employment_type(cls, value: Any) -> str | None:
        return _match_enum(value, EXP_EMPLOYMENT_OPTIONS)

    @field_validator("currently_working_here", mode="before")
    @classmethod
    def _coerce_bool(cls, value: Any) -> bool:
        return value if isinstance(value, bool) else False


class EducationAndExperience(BaseModel):
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)

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
        self.education = [e for e in self.education if e.institute_name or e.degree]
        self.experience = [e for e in self.experience if e.company_name or e.designation]
        return self


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
