"""Corrections applied to LLM profile output using evidence from the resume text."""

from app.profile_schemas import BONCProfileExtraction
from app.services.profile_grounding import ground_profile

ROHAN_TEXT = """ROHAN KUMAR SHARMA
Backend Engineer
CONTACT
+91 90123 45678
rohan.k.example@mail.com
github.com/rohank-example
WORK EXPERIENCE
Senior Backend Engineer, ShipFast Logistics, Jaipur
Jan 2021 - Current | Full-time
Scaled order-tracking APIs to 2M requests/day for https://shipfast.example.com/track.
Software Engineer Intern, Paynest, Bengaluru
May 2017 - Jul 2017 | Internship
EDUCATION
M.Tech, BITS Pilani
2022 - Present
"""


def _profile(**sections) -> BONCProfileExtraction:
    return BONCProfileExtraction.model_validate(sections)


def test_three_part_name_split_from_heading_line():
    p = ground_profile(_profile(basic_details={"first_name": "Rohan", "last_name": "Kumar"}), ROHAN_TEXT)
    b = p.basic_details
    assert (b.first_name, b.middle_name, b.last_name) == ("Rohan", "Kumar", "Sharma")


def test_name_left_alone_without_matching_evidence():
    # The model's last name is not on the heading line: do not overwrite it.
    p = ground_profile(_profile(basic_details={"first_name": "Rohan", "last_name": "Verma"}), ROHAN_TEXT)
    assert (p.basic_details.middle_name, p.basic_details.last_name) == (None, "Verma")
    # No first name: never guess one from headings.
    p = ground_profile(_profile(basic_details={}), ROHAN_TEXT)
    assert p.basic_details.first_name is None
    # Already-correct names stay as the model wrote them.
    p = ground_profile(_profile(basic_details={"first_name": "Priya", "last_name": "Nair"}), "Priya Nair\nDesigner")
    assert (p.basic_details.first_name, p.basic_details.middle_name, p.basic_details.last_name) == ("Priya", None, "Nair")


def test_placeholder_link_repaired_and_missing_profiles_added():
    text = ROHAN_TEXT + "Portfolio: dribbble.com/rohan-designs\nlinkedin.com\n"
    p = ground_profile(
        _profile(networking={"social_links": [
            {"platform": "LinkedIn", "url": "https://.../rohank-example"},
            {"platform": "Website", "url": "https://.../nothing-like-this"},
        ]}),
        text,
    )
    links = [(s.platform, s.url) for s in p.networking.social_links]
    assert links == [
        ("GitHub", "https://github.com/rohank-example"),
        ("Website", "https://dribbble.com/rohan-designs"),
    ]
    # Email domains, company URLs and bare "linkedin.com" are not profiles.
    assert not any("mail.com" in u or "shipfast" in u for _, u in links)


def test_model_links_kept_and_deduplicated():
    p = ground_profile(
        _profile(networking={"social_links": [
            {"platform": "GitHub", "url": "https://www.github.com/rohank-example/"},
            {"platform": "GitHub", "url": "github.com/rohank-example"},
            {"platform": "Website", "url": "https://rohan.dev"},
        ]}),
        ROHAN_TEXT,
    )
    # Links found in the resume take its exact spelling; others are kept as the model gave them.
    assert [s.url for s in p.networking.social_links] == ["https://github.com/rohank-example", "https://rohan.dev"]


def test_ongoing_roles_and_studies_flagged_from_date_ranges():
    p = ground_profile(
        _profile(education_and_experience={
            "experience": [
                {"company_name": "ShipFast Logistics", "from_date": "Jan 2021"},
                {"company_name": "Paynest", "from_date": "May 2017"},
                {"company_name": "Codewave Labs", "from_date": "2018"},  # not in the text
            ],
            "education": [{"institute_name": "BITS Pilani", "from_date": "2022"}],
        }),
        ROHAN_TEXT,
    )
    exp = p.education_and_experience.experience
    assert [e.currently_working_here for e in exp] == [True, False, False]
    assert p.education_and_experience.education[0].currently_studying_here is True


def test_ended_role_with_to_date_is_not_flagged():
    p = ground_profile(
        _profile(education_and_experience={"experience": [
            {"company_name": "ShipFast Logistics", "from_date": "Jan 2021", "to_date": "Mar 2023"},
        ]}),
        ROHAN_TEXT,
    )
    assert p.education_and_experience.experience[0].currently_working_here is False


def test_only_the_entrys_own_date_range_counts():
    # Paynest's own range has ended; a later "2022 - Present" belongs to another section.
    text = "Paynest, Bengaluru\nMay 2017 - Jul 2017\nEDUCATION\nSome College\n2022 - Present\n"
    p = ground_profile(_profile(education_and_experience={"experience": [{"company_name": "Paynest"}]}), text)
    assert p.education_and_experience.experience[0].currently_working_here is False
    text = "Acme Corp\n08/2016 - 06/2018\nGlobex\n2019 – Present\n"
    p = ground_profile(_profile(education_and_experience={"experience": [{"company_name": "Acme Corp"}]}), text)
    assert p.education_and_experience.experience[0].currently_working_here is False


def test_misspelt_link_replaced_by_the_resumes_link_on_the_same_site():
    text = "Priya Nair\nlinkedin.com/in/priya-nair-example\n"
    p = ground_profile(
        _profile(networking={"social_links": [{"platform": "LinkedIn", "url": "https://linkedin.com/in/prinya-nair-example"}]}),
        text,
    )
    assert [s.url for s in p.networking.social_links] == ["https://linkedin.com/in/priya-nair-example"]
