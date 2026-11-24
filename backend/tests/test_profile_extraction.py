"""Tests for BONC resume profile extraction schema, service, and API endpoint."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container
from app.errors import LLMGenerationError, ModelUnavailableError
from app.main import create_app
from app.profile_schemas import BONCProfileExtraction
from app.services.analysis_service import AnalysisService
from app.services.chunker import MarkdownChunker
from app.services.document_parser import DocumentParser, PyPdfTextStrategy
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService
from app.services.profile_extractor import ProfileExtractionService, parse_profile_json
from app.services.summarizer import DocumentSummarizer, SummaryJobManager
from tests.conftest import StubLLM, StubVectorStore, encrypt_pdf, make_text_pdf

RESUME_PAGES = [
    [
        "AARAV MEHTA",
        "Senior Software Engineer | Gurgaon, India",
        "Phone: +91 98765 43210 | LinkedIn: https://linkedin.com/in/aarav-mehta",
        "SUMMARY",
        "Backend engineer with 6 years building Python APIs and payment systems.",
        "EXPERIENCE",
        "Nimbus Commerce - Senior Software Engineer (Full Time), Gurgaon (Apr 2021 - Present)",
        "Led payments and invoicing microservices.",
        "EDUCATION",
        "Delhi Technological University - B.Tech in Computer Engineering (2013 - May 2017)",
    ]
]

SAMPLE_LLM_JSON = {
    "basic_details": {
        "first_name": "Aarav",
        "middle_name": None,
        "last_name": "Mehta",
        "alternate_phone_number": "+91 98765 43210",
        "gender": None,
        "primary_address": {
            "house_flat_building": None,
            "street_area_locality": None,
            "landmark": None,
            "country": "India",
            "state": None,
            "city": "Gurgaon",
            "pincode": None,
        },
        "additional_addresses": [],
    },
    "networking": {
        "profession": "Senior Software Engineer",
        "location": "Gurgaon, India",
        "date_of_birth": None,
        "tag_line": None,
        "about": "Backend engineer with 6 years building Python APIs and payment systems.",
        "social_links": [
            {"platform": "linkedin", "url": "https://linkedin.com/in/aarav-mehta"},
        ],
    },
    "education_and_experience": {
        "education": [
            {
                "institute_name": "Delhi Technological University",
                "degree": "B.Tech",
                "field_of_study": "Computer Engineering",
                "location": None,
                "from_date": "2013",
                "to_date": "May 2017",
                "currently_studying_here": False,
                "description": None,
            }
        ],
        "experience": [
            {
                "company_name": "Nimbus Commerce",
                "designation": "Senior Software Engineer",
                "employment_type": "Full Time",
                "location": "Gurgaon",
                "from_date": "Apr 2021",
                "to_date": "Present",
                "currently_working_here": False,
                "description": "Led payments and invoicing microservices.",
            }
        ],
    },
    "career_preferences": {
        "mode": None,
        "job_category": None,
        "employment_type": None,
        "work_mode": None,
        "preferred_country": None,
        "preferred_state": None,
        "preferred_city": None,
        "experience_years_min": None,
        "experience_years_max": None,
        "salary_inr_per_year_min": None,
        "salary_inr_per_year_max": None,
    },
}


class FakeOffice:
    def __init__(self, pdf_bytes: bytes) -> None:
        self.pdf_bytes = pdf_bytes

    def available_backends(self) -> list[str]:
        return ["fake"]

    def to_pdf(self, src, dst, password=None) -> None:
        dst.write_bytes(self.pdf_bytes)


class UnavailableLLM(StubLLM):
    async def ensure_ready(self) -> None:
        raise ModelUnavailableError("No local model configured.")


def build_test_container(
    tmp_path,
    *,
    llm: StubLLM | None = None,
    max_resume_upload_mb: int = 10,
    office_pdf: bytes | None = None,
) -> Container:
    settings = Settings(
        data_dir=tmp_path,
        max_resume_upload_mb=max_resume_upload_mb,
        llm_load_on_startup=False,
    )
    settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    registry = DocumentRegistry(settings.documents_dir)
    office = FakeOffice(office_pdf) if office_pdf is not None else None
    parser = DocumentParser([PyPdfTextStrategy()], office_converter=office)
    vector_store = StubVectorStore()
    active_llm = llm or StubLLM(reply=json.dumps(SAMPLE_LLM_JSON))
    chunker = MarkdownChunker(target_tokens=100, max_tokens=200, min_tokens=5)
    return Container(
        settings=settings,
        registry=registry,
        parser=parser,
        vector_store=vector_store,
        llm=active_llm,
        ingestion=IngestionService(registry, parser, chunker, vector_store),
        analysis=AnalysisService(vector_store, active_llm, settings),
        summaries=SummaryJobManager(DocumentSummarizer(vector_store, active_llm, settings), registry),
        profile_extractor=ProfileExtractionService(parser, active_llm, settings),
    )


@pytest.fixture
def env(tmp_path):
    resume_pdf = make_text_pdf(RESUME_PAGES)
    container = build_test_container(tmp_path, office_pdf=resume_pdf)
    with TestClient(create_app(container=container)) as client:
        yield client, container, resume_pdf


# ---------------------------------------------------------------------------
# Unit tests: schema & JSON parsing
# ---------------------------------------------------------------------------
def test_empty_profile_keeps_missing_fields_null_and_empty():
    for raw in ({}, {"basic_details": None, "networking": None, "education_and_experience": None, "career_preferences": None}):
        profile = BONCProfileExtraction.model_validate(raw)
        assert profile.basic_details.first_name is None
        assert profile.basic_details.gender is None
        assert profile.basic_details.additional_addresses == []
        assert profile.networking.date_of_birth is None
        assert profile.networking.social_links == []
        assert profile.education_and_experience.education == []
        assert profile.education_and_experience.experience == []
        assert profile.career_preferences.mode is None
        assert profile.career_preferences.job_category is None
        assert profile.career_preferences.work_mode is None
        assert profile.career_preferences.employment_type is None
        assert profile.career_preferences.salary_inr_per_year_min is None


def test_parse_profile_json_handles_fences_wrappers_and_invalid_json():
    fenced = 'Here is the extracted profile:\n```json\n{"profile": {"basic_details": {"first_name": "Aarav",}}}\n```'
    parsed = parse_profile_json(fenced)
    assert parsed["basic_details"]["first_name"] == "Aarav"

    wrapped = '{"data": {"networking": {"profession": "Designer"}}}'
    assert parse_profile_json(wrapped)["networking"]["profession"] == "Designer"

    with pytest.raises(LLMGenerationError):
        parse_profile_json("Sorry, I cannot process this.")


def test_enum_matching_partial_dates_and_ongoing_roles():
    payload = {
        "basic_details": {
            "gender": "Unknown",
            "additional_addresses": [{}, {"city": "Pune"}],
        },
        "networking": {
            "social_links": [
                {"platform": "github", "url": "https://github.com/aarav"},
                {"platform": "Portfolio", "url": "https://aarav.dev"},
                {"platform": "LinkedIn", "url": ""},
            ]
        },
        "education_and_experience": {
            "education": [
                {"institute_name": None, "degree": None},
                {
                    "institute_name": "IIT Delhi",
                    "degree": "B.Tech",
                    "from_date": "2013",
                    "to_date": "May 2017",
                    "currently_studying_here": False,
                },
            ],
            "experience": [
                {"company_name": None, "designation": None},
                {
                    "company_name": "Nimbus",
                    "designation": "Engineer",
                    "employment_type": "Full Time",
                    "from_date": "Apr 2021",
                    "to_date": "Present",
                    "currently_working_here": False,
                },
            ],
        },
        "career_preferences": {
            "job_category": "Spacecraft",
            "employment_type": "Full-time",
            "work_mode": "onsite",
            "experience_years_min": "-2",
            "salary_inr_per_year_min": "1200000",
        },
    }
    profile = BONCProfileExtraction.model_validate(payload)
    assert profile.basic_details.gender is None
    assert len(profile.basic_details.additional_addresses) == 1
    assert profile.basic_details.additional_addresses[0].city == "Pune"

    assert [(s.platform, s.url) for s in profile.networking.social_links] == [
        ("GitHub", "https://github.com/aarav"),
        ("Other", "https://aarav.dev"),
    ]

    assert len(profile.education_and_experience.education) == 1
    edu = profile.education_and_experience.education[0]
    assert edu.from_date == "2013" and edu.to_date == "May 2017" and not edu.currently_studying_here

    assert len(profile.education_and_experience.experience) == 1
    exp = profile.education_and_experience.experience[0]
    assert exp.employment_type == "Full-time"
    assert exp.currently_working_here is True and exp.to_date is None

    assert profile.career_preferences.job_category is None
    assert profile.career_preferences.employment_type == "Full-Time"
    assert profile.career_preferences.work_mode == "On-Site"
    assert profile.career_preferences.experience_years_min is None
    assert profile.career_preferences.salary_inr_per_year_min == 1200000


def test_long_resume_is_trimmed_to_context_window(tmp_path):
    long_pages = [["Aarav Mehta", "Python Engineer"] + ["Built distributed services at scale. " * 20] * 40]
    pdf_path = tmp_path / "long_resume.pdf"
    pdf_path.write_bytes(make_text_pdf(long_pages))

    llm = StubLLM(reply=json.dumps(SAMPLE_LLM_JSON))
    llm.context_window = 2048
    settings = Settings(data_dir=tmp_path, profile_max_new_tokens=300, llm_load_on_startup=False)
    service = ProfileExtractionService(DocumentParser([PyPdfTextStrategy()]), llm, settings)

    import asyncio

    result = asyncio.run(service.extract_from_file(pdf_path))
    assert result.basic_details.first_name == "Aarav"
    prompt_tokens = llm.count_tokens(llm.render_prompt(llm.calls[-1]))
    assert prompt_tokens <= llm.context_window - settings.profile_max_new_tokens


# ---------------------------------------------------------------------------
# API & pipeline isolation tests
# ---------------------------------------------------------------------------
def test_extract_resume_pdf_and_pipeline_isolation(env):
    client, container, resume_pdf = env
    res = client.post(
        "/api/profile/extract-resume",
        files={"file": ("aarav_resume.pdf", resume_pdf, "application/pdf")},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["basic_details"]["first_name"] == "Aarav"
    assert body["basic_details"]["last_name"] == "Mehta"
    assert body["basic_details"]["gender"] is None
    assert body["networking"]["social_links"] == [
        {"platform": "LinkedIn", "url": "https://linkedin.com/in/aarav-mehta"}
    ]
    assert body["education_and_experience"]["experience"][0]["employment_type"] == "Full-time"
    assert body["education_and_experience"]["experience"][0]["currently_working_here"] is True
    assert body["education_and_experience"]["experience"][0]["to_date"] is None
    assert body["career_preferences"]["mode"] is None
    assert body["career_preferences"]["job_category"] is None

    # Verify LLM saw the parsed resume content inside <resume_text>
    user_prompt = container.llm.calls[-1][1].content
    assert "<resume_text>" in user_prompt and "Nimbus Commerce" in user_prompt

    # Verify zero chunking, embedding, ChromaDB, registry, or temp file persistence
    assert container.registry.list() == []
    assert container.vector_store.chunks == {}
    assert list(container.settings.tmp_dir.iterdir()) == []


def test_extract_resume_docx_and_doc(env):
    client, container, _ = env

    docx_buf = io.BytesIO()
    with zipfile.ZipFile(docx_buf, "w") as archive:
        archive.writestr("word/document.xml", "<w:document/>")
    doc_bytes = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64 + "WordDocument".encode("utf-16-le")

    for name, payload, mime in (
        ("resume.docx", docx_buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("resume.doc", doc_bytes, "application/msword"),
    ):
        res = client.post("/api/profile/extract-resume", files={"file": (name, payload, mime)})
        assert res.status_code == 200, res.text
        assert res.json()["basic_details"]["first_name"] == "Aarav"

    assert list(container.settings.tmp_dir.iterdir()) == []


def test_extract_resume_rejects_unsupported_types_and_images(env):
    client, _, _ = env
    res_txt = client.post("/api/profile/extract-resume", files={"file": ("resume.txt", b"hello", "text/plain")})
    assert res_txt.status_code == 415
    assert res_txt.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    from PIL import Image

    img_buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(img_buf, format="PNG")
    res_img = client.post(
        "/api/profile/extract-resume", files={"file": ("scan.png", img_buf.getvalue(), "image/png")}
    )
    assert res_img.status_code == 415
    assert res_img.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_extract_resume_empty_and_oversized_file(tmp_path):
    container = build_test_container(tmp_path, max_resume_upload_mb=1)
    with TestClient(create_app(container=container)) as client:
        res_empty = client.post(
            "/api/profile/extract-resume",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert res_empty.status_code == 422
        assert res_empty.json()["error"]["code"] == "DOCUMENT_UNREADABLE"

        oversized = b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 1024)
        res_big = client.post(
            "/api/profile/extract-resume",
            files={"file": ("huge.pdf", oversized, "application/pdf")},
        )
        assert res_big.status_code == 413
        assert res_big.json()["error"]["code"] == "FILE_TOO_LARGE"
        assert list(container.settings.tmp_dir.iterdir()) == []


def test_extract_resume_encrypted_and_blank_pdf(env):
    client, _, resume_pdf = env
    locked = encrypt_pdf(resume_pdf, "secret")
    res_locked = client.post(
        "/api/profile/extract-resume",
        files={"file": ("locked.pdf", locked, "application/pdf")},
    )
    assert res_locked.status_code == 422
    assert res_locked.json()["error"]["code"] == "PASSWORD_REQUIRED"

    blank = make_text_pdf([[]])
    res_blank = client.post(
        "/api/profile/extract-resume",
        files={"file": ("blank.pdf", blank, "application/pdf")},
    )
    assert res_blank.status_code == 422
    assert res_blank.json()["error"]["code"] == "DOCUMENT_UNREADABLE"


def test_extract_resume_llm_unavailable_and_malformed_json(tmp_path):
    resume_pdf = make_text_pdf(RESUME_PAGES)

    c_unavailable = build_test_container(tmp_path / "u", llm=UnavailableLLM())
    with TestClient(create_app(container=c_unavailable)) as client:
        res = client.post(
            "/api/profile/extract-resume",
            files={"file": ("resume.pdf", resume_pdf, "application/pdf")},
        )
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"

    c_bad_json = build_test_container(tmp_path / "b", llm=StubLLM(reply="Not valid JSON output."))
    with TestClient(create_app(container=c_bad_json)) as client:
        res = client.post(
            "/api/profile/extract-resume",
            files={"file": ("resume.pdf", resume_pdf, "application/pdf")},
        )
        assert res.status_code == 500
        assert res.json()["error"]["code"] == "GENERATION_FAILED"


def test_profile_html_served_and_linked(env):
    client, _, _ = env
    res_profile = client.get("/profile.html")
    assert res_profile.status_code == 200
    assert "/api/profile/extract-resume" in res_profile.text

    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "/profile.html" in res_index.text
