"""Corrections applied to LLM profile output using evidence from the resume text."""

from app.profile_schemas import BONCProfileExtraction
from app.services.profile_grounding import ground_profile, same_phone_number

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


def test_same_phone_number_ignores_formatting_and_country_prefix():
    for other in ("+91 90123 45678", "9012345678", "09012345678", "0091-90123-45678", "(+91) 901 234 5678"):
        assert same_phone_number("+91 90123 45678", other), other
    assert not same_phone_number("+91 90123 45678", "+91 90123 45679")
    assert not same_phone_number("+91 90123 45678", None)
    assert not same_phone_number(None, None)
    assert not same_phone_number("123", "123")  # too short to be a phone number


def test_alternate_phone_cleared_when_it_is_the_primary_number():
    phone = {"first_name": "Rohan", "alternate_phone_number": "+91 90123 45678"}
    p = ground_profile(_profile(basic_details=phone), ROHAN_TEXT, primary_phone="9012345678")
    assert p.basic_details.alternate_phone_number is None
    # A different primary number, or none at all, keeps the resume's number.
    p = ground_profile(_profile(basic_details=phone), ROHAN_TEXT, primary_phone="+91 99999 00000")
    assert p.basic_details.alternate_phone_number == "+91 90123 45678"
    p = ground_profile(_profile(basic_details=phone), ROHAN_TEXT)
    assert p.basic_details.alternate_phone_number == "+91 90123 45678"
    # Partials can arrive before any resume text is known; the check still applies.
    p = ground_profile(_profile(basic_details=phone), "", primary_phone="09012345678")
    assert p.basic_details.alternate_phone_number is None


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
    # Links found in the resume take its exact spelling; ones it does not contain (the model
    # invents "<name>.dev" or "linkedin.com/in/<name>" from the email) are dropped.
    assert [s.url for s in p.networking.social_links] == ["https://github.com/rohank-example"]
    text = ROHAN_TEXT + "Portfolio: rohan.dev\n"
    p = ground_profile(_profile(networking={"social_links": [{"platform": "Website", "url": "https://rohan.dev"}]}), text)
    assert [s.url for s in p.networking.social_links] == ["https://rohan.dev", "https://github.com/rohank-example"]


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


WORK_TEXT = """SKILLS
Python, Django, PostgreSQL, Six Sigma
TOOLS Jira, Git
LANGUAGES English (fluent), Hindi (native)
CERTIFICATIONS AWS Certified Solutions Architect - Associate (2023)
EXPERIENCE
ShipFast Logistics - Senior Backend Engineer
Scaled order-tracking APIs to 2M requests/day.
Cut cloud spend by 30% with autoscaling.
Paynest - Intern
Built reconciliation scripts for UPI transactions.
"""


def test_key_outcomes_keep_only_the_roles_own_bullets():
    p = ground_profile(
        _profile(education_and_experience={"experience": [
            {"company_name": "ShipFast Logistics", "key_outcomes":
                "• Scaled order-tracking APIs to 2M requests/day.\n"
                "• Built reconciliation scripts for UPI transactions.\n"  # belongs to Paynest
                "• Collaborated with stakeholders to ensure delivery excellence\n"  # invented filler
                "- Cut cloud spend by 30% with autoscaling"},
            {"company_name": "Paynest", "key_outcomes": "• Built reconciliation scripts for UPI transactions."},
            {"company_name": "Not In Resume Ltd", "key_outcomes": "• Something"},
        ]}),
        WORK_TEXT,
    )
    exp = p.education_and_experience.experience
    # Read from the role's own lines in the resume: no bullet from Paynest, no invented filler.
    assert exp[0].key_outcomes == "• Scaled order-tracking APIs to 2M requests/day.\n• Cut cloud spend by 30% with autoscaling."
    assert exp[1].key_outcomes == "• Built reconciliation scripts for UPI transactions."
    assert exp[2].key_outcomes is None


def test_skills_tools_languages_and_certifications_must_appear_in_resume():
    p = ground_profile(
        _profile(
            education_and_experience={
                "skills": ["Python", "Kubernetes", "Django", "PostgreSQL", "Leadership", "Six Sigma", "C"],
                "tools": ["Jira", "Confluence", "git"],
                "languages": ["English", "French", "Python", {"language": "Hindi", "proficiency": "native"}],
                "certifications": ["AWS Certified Solutions Architect Associate", "PMP"],
            },
            career_preferences={"target_roles": ["Senior Backend Engineer", "Chief Executive Officer"]},
        ),
        WORK_TEXT,
    )
    ee = p.education_and_experience
    # Invented primary skills are dropped and secondary skills move up to keep five.
    assert [s.name for s in ee.skills.primary] == ["Python", "Django", "PostgreSQL", "Six Sigma", "C"]
    assert ee.skills.secondary == []
    assert ee.tools == ["Jira", "git"]
    assert [(lang.language, lang.proficiency) for lang in ee.languages] == [("English", None), ("Hindi", "Native / Bilingual")]
    assert [c.name for c in ee.certifications] == ["AWS Certified Solutions Architect Associate"]
    assert p.career_preferences.target_roles == ["Senior Backend Engineer"]


def test_middle_initial_expanded_from_heading():
    p = ground_profile(_profile(basic_details={"first_name": "Meera", "middle_name": "A", "last_name": "Iyer"}), "Meera Anand Iyer\nOperations Leader")
    assert (p.basic_details.first_name, p.basic_details.middle_name, p.basic_details.last_name) == ("Meera", "Anand", "Iyer")


def test_split_skill_rejoined_only_when_written_as_one_phrase():
    text = "SKILLS\nSix Sigma, Agentic AI Workflows, KPI Reporting\nPython\nDjango\n"
    p = ground_profile(
        _profile(education_and_experience={"skills": ["Six Sigma", "Agentic AI", "Workflows", "KPI Reporting", "Python", "Django"]}),
        text,
    )
    s = p.education_and_experience.skills
    assert [x.name for x in s.primary] == ["Six Sigma", "Agentic AI Workflows", "KPI Reporting", "Python", "Django"]
    assert s.secondary == []


def test_certification_issuer_suffix_and_no_expiry():
    text = "CERTIFICATIONS\nLean Six Sigma Black Belt - ASQ (2019, no expiry)\nPMP - Project Management Institute (expires Mar 2025)\n"
    p = ground_profile(
        _profile(education_and_experience={"certifications": [
            {"name": "Lean Six Sigma Black Belt - ASQ", "issuer": "ASQ", "issue_date": "2019"},
            {"name": "PMP - Project Management Institute", "issuer": "Project Management Institute"},
        ]}),
        text,
    )
    certs = p.education_and_experience.certifications
    assert [(c.name, c.does_not_expire) for c in certs] == [("Lean Six Sigma Black Belt", True), ("PMP", False)]



def test_skill_split_by_a_wrapped_list_line_is_rejoined():
    # A comma list long enough to hit the page width, wrapped mid-skill.
    text = "SKILLS\nKPI Reporting, Supply Chain Management, Vendor Management, Agentic AI\nWorkflows\nTOOLS\nSAP\n"
    p = ground_profile(
        _profile(education_and_experience={"skills": ["KPI Reporting", "Agentic AI", "Workflows", "Supply Chain Management"]}), text
    )
    names = [x.name for x in p.education_and_experience.skills.primary]
    # The skills line also tops up the model's short list, without re-adding the split pieces.
    assert names == ["KPI Reporting", "Agentic AI Workflows", "Supply Chain Management", "Vendor Management"]


SECTIONS_TEXT = """ANITA DESAI
Summary
Skilled in production support, troubleshooting, bug fixing and requirement analysis.
Technical Skills
Languages: C#, JavaScript, SQL
Frameworks & Technologies: ASP.NET MVC, Entity Framework,
ADO.NET, jQuery
Developer Tools: Visual Studio, Git, IIS
LLM APIs (OpenAI, LangChain), Delivered secure, fixed bugs
Domains:
Banking, Health Care
PROFESSIONAL EXPERIENCE
Acme Softech, Pune
Software Engineer | Jan 2021 - Present
Built a billing portal.
Technologies: React, Node.js
Language Known
English, Hindi (native), Python
"""


def test_skills_tools_and_languages_read_from_sections_when_the_model_skips_them():
    p = ground_profile(
        _profile(education_and_experience={"experience": [{"company_name": "Acme Softech", "location": "Pune"}]}),
        SECTIONS_TEXT,
    )
    ee = p.education_and_experience
    skills = [s.name for s in ee.skills.primary + ee.skills.secondary]
    assert skills == ["C#", "JavaScript", "SQL", "LLM APIs (OpenAI, LangChain)"]
    assert ee.tools == ["ASP.NET MVC", "Entity Framework", "ADO.NET", "jQuery", "Visual Studio", "Git", "IIS", "React", "Node.js"]
    assert ee.sub_domains == ["Banking", "Health Care"]
    assert [(lang.language, lang.proficiency) for lang in ee.languages] == [("English", None), ("Hindi", "Native / Bilingual")]


def test_section_fallback_never_overrides_the_models_lists():
    p = ground_profile(
        _profile(education_and_experience={
            "skills": [{"name": "SQL", "proficiency": 8}, "JavaScript", "C#", "Node.js", "Python"],
            "tools": ["Git"], "languages": ["English"],
        }),
        SECTIONS_TEXT + "Python\n",
    )
    ee = p.education_and_experience
    assert [s.name for s in ee.skills.primary] == ["SQL", "JavaScript", "C#", "Node.js", "Python"]
    assert ee.skills.secondary == []
    assert ee.tools == ["Git"]
    assert [lang.language for lang in ee.languages] == ["English"]


def test_renamed_or_misplaced_list_keys_are_accepted():
    p = _profile(
        education_and_experience={"technical_skills": ["Python"], "languages_known": ["English", "Hindi"]},
        personal_details={"certificates": ["PMP"]},
    )
    ee = p.education_and_experience
    assert [s.name for s in ee.skills.primary] == ["Python"]
    assert [lang.language for lang in ee.languages] == ["English", "Hindi"]
    assert [c.name for c in ee.certifications] == ["PMP"]


def test_sparse_model_skills_topped_up_from_the_skills_section():
    p = ground_profile(_profile(education_and_experience={"skills": [{"name": "SQL", "proficiency": 8}]}), SECTIONS_TEXT)
    skills = p.education_and_experience.skills.primary
    assert [(s.name, s.proficiency) for s in skills] == [("SQL", 8), ("C#", None), ("JavaScript", None), ("LLM APIs (OpenAI, LangChain)", None)]


MEERA_TEXT = """MEERA JOSHI
.Net Developer
EDUCATION
B.E. (Computer) from Pune University (2012) - 72%
10+2 from Maharashtra Board (2008) - 81%
10th from Maharashtra Board (2006) - 85%
CONTACT DETAILS
ADDRESS : 12 HILL VIEW, COLLEGE ROAD, NASHIK (MAHARASHTRA) -422005
WORK EXPERIENCE
Brightsoft Pvt Ltd / Remote / (May 2022 - Current)
• Designation - Software Developer
Codeline Ltd / Nashik / (Jun 2016 - Apr 2022)
Junior Developer
Webcraft Labs / Mumbai / (Jan 2015 - Jun 2015)
Web Developer Intern
TRAINING & CERTIFICATION:
Skillup Academy (Jan 2022 - Mar 2022) - .Net Full Stack Development
Nimbus Infotech (45 days) - .Net Industrial training (During College)
AWS Developer Associate (2023) - Amazon
SKILLS
.Net, C#, MVC, WebAPI, SQL, .Net Core, JavaScript, jQuery, HTML, Ado.Net
LANGUAGE PROFICIENCY:
Hindi, English (Fluent), Marathi
"""

MEERA_MODEL = {
    "basic_details": {"first_name": "Meera", "last_name": "Joshi",
                      "primary_address": {"city": "Nashik", "state": "Maharashtra"}},
    "education_and_experience": {
        "education": [
            {"institute_name": "Pune University", "degree": "B.E. (Computer)", "field_of_study": "Computer",
             "location": "Nashik (MAHARASHTRA)", "from_date": "2012"},
            {"institute_name": "Maharashtra Board", "degree": "10+2", "field_of_study": "Computer",
             "location": "Nashik (MAHARASHTRA)", "from_date": "2008"},
        ],
        "experience": [
            {"company_name": "Brightsoft Pvt Ltd", "designation": "Software Developer", "location": "Nashik",
             "from_date": "May 2022", "to_date": "Current"},
            {"company_name": "Codeline Ltd", "designation": "Junior Developer", "location": "Nashik",
             "from_date": "Jun 2016", "to_date": "Apr 2022"},
            {"company_name": "Webcraft Labs", "designation": "Web Developer Intern", "location": "Mumbai",
             "from_date": "Jan 2015", "to_date": "Jun 2015"},
        ],
        "skills": [".Net", "C#", "MVC", "WebAPI", "SQL", ".Net Core", "JavaScript", "jQuery", "HTML", "Ado.Net"],
        "languages": [{"language": "Hindi", "proficiency": "Native"}, {"language": "English", "proficiency": "Fluent"}],
        "certifications": [
            {"name": "Skillup Academy (Jan 2022 - Mar 2022) - .Net Full Stack Development"},
            {"name": "Nimbus Infotech (45 days) - .Net Industrial training (During College)"},
            {"name": "AWS Developer Associate (2023) - Amazon"},
        ],
    },
}


def test_frameworks_in_a_single_skills_list_move_to_tools():
    ee = ground_profile(_profile(**MEERA_MODEL), MEERA_TEXT).education_and_experience
    assert [s.name for s in ee.skills.primary] == [".Net", "C#", "MVC", "WebAPI", "SQL"]
    assert [s.name for s in ee.skills.secondary] == ["JavaScript", "HTML"]
    assert ee.tools == [".Net Core", "jQuery", "Ado.Net"]


def test_certification_provider_and_period_split_out_of_the_name():
    certs = ground_profile(_profile(**MEERA_MODEL), MEERA_TEXT).education_and_experience.certifications
    assert [(c.name, c.issuer, c.issue_date) for c in certs] == [
        (".Net Full Stack Development", "Skillup Academy", "Mar 2022"),
        (".Net Industrial training (During College)", "Nimbus Infotech", None),
        ("AWS Developer Associate", "Amazon", "2023"),
    ]
    # The model sometimes fills the issuer but still leaves the whole line in the name.
    model = {"education_and_experience": {"certifications": [
        {"name": "Skillup Academy (Jan 2022 - Mar 2022) - .Net Full Stack Development", "issuer": "Skillup Academy"},
        {"name": "Nimbus Infotech (45 days) - .Net Industrial training (During College)", "issuer": "Other Org"},
    ]}}
    certs = ground_profile(_profile(**model), MEERA_TEXT).education_and_experience.certifications
    assert [(c.name, c.issuer, c.issue_date) for c in certs] == [
        (".Net Full Stack Development", "Skillup Academy", "Mar 2022"),
        ("Nimbus Infotech (45 days) - .Net Industrial training (During College)", "Other Org", None),
    ]


def test_home_city_copied_onto_entries_is_removed():
    ee = ground_profile(_profile(**MEERA_MODEL), MEERA_TEXT).education_and_experience
    assert [e.location for e in ee.education] == [None, None, None]
    # Remote is written on the role's own line; Nashik really is Codeline's city; Mumbai is not home.
    assert [e.location for e in ee.experience] == ["Remote", "Nashik", "Mumbai"]


def test_school_stream_passing_year_employment_type_and_language_level():
    ee = ground_profile(_profile(**MEERA_MODEL), MEERA_TEXT).education_and_experience
    # A school entry does not inherit the degree's field; a lone year is the passing year.
    # The 10th row the model skipped is read from the education section.
    assert [(e.degree, e.field_of_study, e.from_date, e.to_date) for e in ee.education] == [
        ("B.E. (Computer)", "Computer", None, "2012"), ("10+2", None, None, "2008"), ("10th", None, None, "2006"),
    ]
    assert (ee.education[2].institute_name, ee.education[2].description) == ("Maharashtra Board", "85%")
    assert [e.employment_type for e in ee.experience] == [None, None, "Internship"]
    assert ee.experience[0].currently_working_here is True
    # "Native" is never written for Hindi; "Fluent" is for English.
    assert [(lang.language, lang.proficiency) for lang in ee.languages] == [("Hindi", None), ("English", "Fluent")]


def test_education_date_range_is_not_treated_as_a_passing_year():
    text = "EDUCATION\nM.Tech, BITS Pilani\n2018 - 2020\n"
    model = {"education_and_experience": {"education": [{"institute_name": "BITS Pilani", "from_date": "2018"}]}}
    edu = ground_profile(_profile(**model), text).education_and_experience.education[0]
    assert (edu.from_date, edu.to_date) == ("2018", None)


SUBLABEL_TEXT = """KIRAN RAO
CORE SKILLS

Languages & Data
C#, VB.NET, SQL Server, SSIS

Frameworks
ASP.NET MVC, DevExpress, Web API

Practices
Agile/Scrum, Team Leadership, Mentoring

CERTIFICATIONS
• Azure Developer Associate
PROFESSIONAL EXPERIENCE
Senior Software Developer | Acme Pvt. Ltd. Mar 2024 – Present
LANGUAGES
English — Fluent | Hindi — Native
"""


def test_skill_sub_headings_on_their_own_line_are_read_as_labels():
    model = {"education_and_experience": {"experience": [{"company_name": "Acme Pvt. Ltd.", "designation": "Senior Software Developer"}]}}
    ee = ground_profile(_profile(**model), SUBLABEL_TEXT).education_and_experience
    skills = [s.name for s in ee.skills.primary + ee.skills.secondary]
    assert skills == ["C#", "VB.NET", "SQL Server", "SSIS", "Agile/Scrum", "Team Leadership", "Mentoring"]
    assert ee.tools == ["ASP.NET MVC", "DevExpress", "Web API"]
    assert [(lang.language, lang.proficiency) for lang in ee.languages] == [("English", "Fluent"), ("Hindi", "Native / Bilingual")]
    # A software role with a technical skill set gets an industry when the model gave none.
    assert ee.industry_clusters == ["Technology & Software"]


def test_languages_returned_as_a_mapping_are_kept():
    ee = _profile(languages={"English": "Fluent", "Hindi": "Native"}).education_and_experience
    assert [(lang.language, lang.proficiency) for lang in ee.languages] == [("English", "Fluent"), ("Hindi", "Native / Bilingual")]


def test_sub_domains_are_niches_not_skills():
    text = "SKILLS\nPython, SQL, Pandas\nDomains: Payments, Data Analytics\n"
    model = {"education_and_experience": {
        "skills": ["Python", "SQL"], "tools": ["Pandas"],
        "sub_domains": ["Payments", "Python", "SQL", "Pandas", "Data Analytics"],
    }}
    ee = ground_profile(_profile(**model), text).education_and_experience
    assert ee.sub_domains == ["Payments", "Data Analytics"]
    assert ee.industry_clusters == []  # no software role named: the industry is not guessed


def test_middle_name_missing_from_the_heading_line_is_dropped():
    text = "HIMA DESAI\n.Net Developer\nEMAIL: HIMA_JOSHI@EXAMPLE.COM\n"
    p = ground_profile(_profile(basic_details={"first_name": "Hima", "middle_name": "Joshi", "last_name": "Desai"}), text)
    b = p.basic_details
    assert (b.first_name, b.middle_name, b.last_name) == ("Hima", None, "Desai")


ROLES_TEXT = """WORK EXPERIENCE
Brightsoft Pvt Ltd / Remote / (May 2022 - Current)
• Designation – Consultant (Software Developer)
• Project – Atlas - a reporting portal that gives managers self-service
analytics across the organisation. (ASP.NET & SQL)
• Contribution – Reduced report load time by 40% by rewriting the
SQL layer
Codeline Ltd / Pune / (Jun 2016 - Apr 2022)
• Designation - Software Developer
• Client Name - Northwind
• Responsibility - Analysis, Design, Development and Testing
• Contribution – created and updated modules for tagging and hatching
EDUCATION
B.E. from Pune University (2016)
"""


def test_role_description_and_outcomes_are_read_from_the_roles_own_lines():
    # The model is no longer asked for these: they come from each role's own lines.
    model = {"education_and_experience": {"experience": [
        {"company_name": "Brightsoft Pvt Ltd", "designation": "Consultant (Software Developer)"},
        {"company_name": "Codeline Ltd", "designation": "Software Developer"},
    ]}}
    exp = ground_profile(_profile(**model), ROLES_TEXT).education_and_experience.experience
    # Wrapped lines are joined, "Designation –" lines dropped, and a measured result is the outcome.
    assert exp[0].key_outcomes == "• Contribution – Reduced report load time by 40% by rewriting the SQL layer"
    assert exp[0].description == (
        "Project – Atlas - a reporting portal that gives managers self-service analytics across the organisation. (ASP.NET & SQL)"
    )
    # Another role's title inside a bullet does not end the role; duties stay in the description.
    assert exp[1].key_outcomes == "• Contribution – created and updated modules for tagging and hatching"
    assert exp[1].description == "Client Name - Northwind\nResponsibility - Analysis, Design, Development and Testing"


def test_autofill_fills_at_most_six_secondary_skills():
    # Product decision: autofill stops at 5 primary + 6 secondary; the user adds the rest by hand.
    names = ["Python", "SQL", "Java", "C#", "Go", "Kotlin", "Swift", "Ruby", "PHP", "Rust", "Scala", "Perl", "Dart"]
    text = "SKILLS\n" + ", ".join(names) + "\n"
    ee = ground_profile(_profile(education_and_experience={"skills": names}), text).education_and_experience
    assert [s.name for s in ee.skills.primary] == names[:5]
    assert [s.name for s in ee.skills.secondary] == names[5:11]


def test_certificate_dates_need_their_own_line_and_a_course_is_issued_at_its_end():
    model = {"education_and_experience": {"certifications": [
        {"name": ".Net Full Stack Development", "issuer": "Skillup Academy", "issue_date": "Jan 2022"},  # the period's start
        {"name": ".Net Industrial training", "issuer": "Nimbus Infotech", "issue_date": "Jan 2022"},  # copied, no date on its line
        {"name": "AWS Developer Associate", "issuer": "Amazon"},  # dated "(2023)" on its line
    ]}}
    certs = ground_profile(_profile(**model), MEERA_TEXT).education_and_experience.certifications
    assert [c.issue_date for c in certs] == ["Mar 2022", None, "2023"]


def test_employment_type_stated_on_the_role_overrides_the_models_default():
    model = {"education_and_experience": {"experience": [
        {"company_name": "Codeline Ltd", "designation": "Junior Developer", "employment_type": "Full-time"},
        {"company_name": "Webcraft Labs", "designation": "Web Developer Intern", "employment_type": "Full-time"},
    ]}}
    exp = ground_profile(_profile(**model), MEERA_TEXT).education_and_experience.experience
    # Nothing on Codeline's lines contradicts the model; "Intern" in Webcraft's title does.
    assert [e.employment_type for e in exp] == ["Full-time", "Internship"]


def test_profession_from_title_line_else_current_role():
    text = "## Ilze Berzkalne\n\n## Software Tester\n\nilze@example.com · +1 555-0167\n"
    p = ground_profile(_profile(basic_details={"first_name": "Ilze", "last_name": "Berzkalne"}), text)
    assert p.networking.profession == "Software Tester"
    # A contact line under the name is not a title: use the current role instead.
    p = ground_profile(
        _profile(
            basic_details={"first_name": "Rohan", "last_name": "Sharma"},
            education_and_experience={"experience": [
                {"company_name": "Paynest", "designation": "Intern", "to_date": "2017"},
                {"company_name": "ShipFast", "designation": "Senior Backend Engineer", "to_date": "Present"},
            ]},
        ),
        "Rohan Sharma\n+91 90123 45678 | Jaipur\n",
    )
    assert p.networking.profession == "Senior Backend Engineer"
    # The model's own value is kept.
    p = ground_profile(_profile(networking={"profession": "Designer"}, basic_details={"first_name": "Ilze"}), text)
    assert p.networking.profession == "Designer"


def test_work_modes_only_from_a_stated_preference():
    prefs = "Languages\nEnglish\nPreferences\nRemote, Hybrid · Part-time\nSummary\nTester."
    p = ground_profile(_profile(basic_details={"first_name": "Ilze"}), prefs)
    assert p.career_preferences.work_modes == ["Remote", "Hybrid"]
    assert p.career_preferences.work_mode == "Remote"
    p = ground_profile(_profile(basic_details={"first_name": "Ilze"}), "Work mode: On-site\n")
    assert p.career_preferences.work_modes == ["On-Site"]
    # A job located "Remote" is not a preference.
    p = ground_profile(_profile(basic_details={"first_name": "Ilze"}), "Experience\nQA Engineer, Acme\nRemote\n2021 - 2023\n")
    assert p.career_preferences.work_modes == []


def test_company_name_rejoined_across_wrapped_ocr_table_cells():
    text = (
        "| EXPERIENCE | | SKILLS |\n|---|\n"
        "| Senior Database Administrator — Baltic | Dec 2023 - Jan | • Docker |\n"
        "| Vault Data Services | 2026 | • Gitlab |\n"
        "| Cloud Platform Engineer \\ Halberstadt | | Dec 2023 - Jan |\n|---|\n| Datawerk | | 2025 |\n"
    )
    p = ground_profile(_profile(education_and_experience={"experience": [
        {"company_name": "Vault Data Services", "designation": "Senior Database Administrator"},
        {"company_name": "Datawerk", "designation": "Cloud Platform Engineer"},
    ]}), text)
    assert [e.company_name for e in p.education_and_experience.experience] == [
        "Baltic Vault Data Services", "Halberstadt Datawerk",
    ]


def test_email_written_as_link_is_dropped():
    p = ground_profile(_profile(networking={"social_links": [{"platform": "Website", "url": "https://roscoe.h@example.com"}]}), "Roscoe\nroscoe.h@example.com")
    assert p.networking.social_links == []


def test_links_not_in_the_resume_are_dropped():
    text = "Nathaniel Pemberton\nnathaniel.pemberton@example.com\ngithub.com/npemberton\n"
    p = ground_profile(_profile(networking={"social_links": [
        {"platform": "LinkedIn", "url": "https://www.linkedin.com/in/nathaniel-pemberton"},  # invented
        {"platform": "Website", "url": "https://nathaniel-pemberton.com"},  # invented
        {"platform": "GitHub", "url": "https://github.com/npemberton"},
    ]}), text)
    assert [l.url for l in p.networking.social_links] == ["https://github.com/npemberton"]


def test_work_modes_in_ocr_table_cells_and_with_trailing_terms():
    p = ground_profile(_profile(), "PREFERENCES\n\n| Hybrid · Part-time · Project- |\n|---|\n| based |\n")
    assert p.career_preferences.work_modes == ["Hybrid"]
    p = ground_profile(_profile(), "PREFERENCES\n\nRemote Permanent\n\n## HOBBIES\n")
    assert p.career_preferences.work_modes == ["Remote"]


def test_projects_listed_as_jobs_and_type_as_title_are_fixed():
    text = (
        "EXPERIENCE\n\nWhiteCore Consulting JUNE 2023 – PRESENT\n\n"
        "PROJECTS\nFINTRIX\nTracks expenses.\nTRUSTLEDGER\nA ledger app.\n\nEDUCATION\nBITS Pilani\n"
    )
    p = ground_profile(_profile(education_and_experience={"experience": [
        {"company_name": "WhiteCore Consulting", "designation": "Full-time", "from_date": "JUNE 2023", "to_date": "Present"},
        {"company_name": "TRUSTLEDGER", "designation": "Full-time", "from_date": "JUNE 2023"},
    ]}), text)
    exp = p.education_and_experience.experience
    assert [(e.company_name, e.designation, e.employment_type) for e in exp] == [("WhiteCore Consulting", None, "Full-time")]


def test_company_spelling_taken_from_the_resume():
    text = "Experience\nDot NET Developer Devlabs Alliance\nJan 2020 - Mar 2021\nEducation\nAmity University\n"
    p = ground_profile(_profile(education_and_experience={
        "experience": [{"company_name": "Dowlabs Alliance", "designation": "Dot NET Developer"}],
        "education": [{"institute_name": "Amity Universty"}],
    }), text)
    ee = p.education_and_experience
    assert ee.experience[0].company_name == "Devlabs Alliance"
    assert ee.education[0].institute_name == "Amity University"
    # Nothing close enough in the text: the model's spelling stays.
    p = ground_profile(_profile(education_and_experience={"experience": [{"company_name": "Zenith Labs"}]}), text)
    assert p.education_and_experience.experience[0].company_name == "Zenith Labs"


def test_address_from_the_contact_line():
    cases = {
        "NEHA S\nFull Stack Developer: C# | : Coimbatore, Tamil Nadu :+91 9000022222 | :m@example.com\n":
            {"city": "Coimbatore", "state": "Tamil Nadu", "country": "India"},
        "VIKRAM R\n+91 9000033333 | k@example.com | Coimbatore, India | github.com/x\n":
            {"city": "Coimbatore", "state": None, "country": "India"},
        "Sneha K\nAurangabad, MH, India | +91 9000044444\n":
            {"city": "Aurangabad", "state": "Maharashtra", "country": "India"},
        "Rahul Bedi\nrahul@example.com 9000055555 | Mohali, Punjab, 160055\n":
            {"city": "Mohali", "state": "Punjab", "country": "India", "pincode": "160055"},
    }
    for text, want in cases.items():
        # The model put the city in the street field and wrote the state abbreviation.
        p = ground_profile(_profile(basic_details={"primary_address": {"street_area_locality": "Aurangabad", "state": "MH"}}
                                    if "Aurangabad" in text else {}), text)
        a = p.basic_details.primary_address
        assert {k: getattr(a, k) for k in want} == want, text
        assert a.street_area_locality is None
    # A skills line is not a place; a different city the model read is kept.
    p = ground_profile(_profile(), "Ravi\nSkills: Python, Django\n")
    assert p.basic_details.primary_address.city is None
    p = ground_profile(_profile(basic_details={"primary_address": {"city": "Pune"}}), "Ravi\nNoida, Uttar Pradesh\n")
    assert p.basic_details.primary_address.city == "Pune"


def test_project_sub_label_inside_a_job_is_not_a_projects_section():
    text = (
        "WORK EXPERIENCE\nSr. Developer Absolute ERP\n07/2024 - Present\nProject\nERP modules.\n"
        "SKILLS\nC#\nSoftware Engineer Code Aspire\n2019 - 2021\nProject\nA CRM.\nEDUCATION\nAKTU\n"
    )
    p = ground_profile(_profile(education_and_experience={"experience": [
        {"company_name": "Absolute ERP", "designation": "Sr. Developer"},
        {"company_name": "Code Aspire", "designation": "Software Engineer"},
    ]}), text)
    assert [e.company_name for e in p.education_and_experience.experience] == ["Absolute ERP", "Code Aspire"]
