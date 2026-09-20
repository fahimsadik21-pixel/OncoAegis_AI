"""Source-aware educational cancer information service.

The service deliberately separates general education from specialist medical
analysis. A configured language model may improve wording, but the fallback is
useful without an external model and always carries the same safety boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import uuid
from dotenv import load_dotenv

from app.registry.cancer_registry import get_cancer_capability
from app.schemas.cancer_information import (
    CancerInformationRequest,
    CancerInformationResponse,
    InformationSection,
    InformationSource,
)

load_dotenv()

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_BENGALI = re.compile(r"[\u0980-\u09ff]")
_PERSONAL_OR_DIAGNOSTIC = re.compile(
    r"\b(do i have|is this cancer|what stage|my scan|my report|diagnos|আমার|আমাকে|স্টেজ|ক্যান্সার হয়েছে)\b",
    re.IGNORECASE,
)
_URGENT_TERMS = re.compile(
    r"\b(emergency|urgent|severe bleeding|can't breathe|cannot breathe|confusion|fainting|uncontrolled pain|high fever|chemotherapy fever)\b|জরুরি|শ্বাসকষ্ট|অতিরিক্ত রক্তপাত|অজ্ঞান|খুব বেশি ব্যথা|জ্বর",
    re.IGNORECASE,
)


_PERSONAL_OR_DIAGNOSTIC = re.compile(
    r"\b(do i have|is this cancer|what stage|my scan|my report|diagnos)\b|আমার|আমাকে|স্টেজ|ক্যান্সার হয়েছে",
    re.IGNORECASE,
)
_URGENT_TERMS = re.compile(
    r"\b(emergency|urgent|severe bleeding|can't breathe|cannot breathe|confusion|fainting|uncontrolled pain|high fever|chemotherapy fever)\b|জরুরি|শ্বাসকষ্ট|অতিরিক্ত রক্তপাত|অজ্ঞান|খুব বেশি ব্যথা|জ্বর",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CancerTopic:
    key: str
    label: str
    registry_key: str | None


_ALIASES = {
    "lung": ("lung_cancer", "Lung cancer"),
    "lung cancer": ("lung_cancer", "Lung cancer"),
    "breast": ("breast_cancer", "Breast cancer"),
    "breast cancer": ("breast_cancer", "Breast cancer"),
    "brain": ("brain_cancer", "Brain cancer"),
    "brain tumor": ("brain_cancer", "Brain cancer"),
    "liver": ("liver_cancer", "Liver cancer"),
    "liver cancer": ("liver_cancer", "Liver cancer"),
    "pancreas": ("pancreatic_cancer", "Pancreatic cancer"),
    "pancreatic cancer": ("pancreatic_cancer", "Pancreatic cancer"),
    "colon": ("colon_cancer", "Colon cancer"),
    "colon cancer": ("colon_cancer", "Colon cancer"),
    "colorectal": ("colorectal_cancer", "Colorectal cancer"),
    "colorectal cancer": ("colorectal_cancer", "Colorectal cancer"),
    "skin": ("skin_cancer", "Skin cancer"),
    "skin cancer": ("skin_cancer", "Skin cancer"),
    "thyroid": ("thyroid_cancer", "Thyroid cancer"),
    "thyroid cancer": ("thyroid_cancer", "Thyroid cancer"),
    "blood cancer": ("blood_cancer", "Blood cancer"),
    "leukemia": ("blood_cancer", "Leukemia"),
    "all": ("acute_lymphoblastic_leukemia", "Acute lymphoblastic leukemia"),
    "aml": ("acute_myeloid_leukemia", "Acute myeloid leukemia"),
}

# Topic-aware educational anchors for the offline fallback. These are deliberately
# high-level; the configured provider can expand them with the user's exact intent.
_TOPIC_CONTEXT = {
    "breast_cancer": ("Breast cancer starts in breast tissue; subtype, hormone receptors, HER2 status and stage strongly affect evaluation and care.", "স্তন ক্যান্সার স্তনের টিস্যুতে শুরু হয়; subtype, hormone receptor, HER2 status এবং stage অনুযায়ী মূল্যায়ন ও চিকিৎসা বদলে যায়।"),
    "lung_cancer": ("Lung cancer includes different types such as non-small-cell and small-cell cancer; smoking history, molecular markers, stage and lung function are important context.", "ফুসফুসের ক্যান্সারের বিভিন্ন ধরন আছে, যেমন non-small-cell ও small-cell cancer; ধূমপানের ইতিহাস, molecular marker, stage এবং ফুসফুসের কার্যক্ষমতা গুরুত্বপূর্ণ।"),
    "colorectal_cancer": ("Colorectal cancer affects the colon or rectum; bowel changes, bleeding, location, stage and molecular findings guide evaluation.", "কোলোরেক্টাল ক্যান্সার colon বা rectum-এ হয়; পায়খানার পরিবর্তন, রক্তপাত, অবস্থান, stage ও molecular finding মূল্যায়নে গুরুত্বপূর্ণ।"),
    "colon_cancer": ("Colon cancer affects the large bowel; symptoms, screening history, pathology, stage and molecular findings guide next steps.", "কোলন ক্যান্সার বৃহদান্ত্রে হয়; উপসর্গ, screening history, pathology, stage ও molecular finding পরবর্তী পদক্ষেপ নির্ধারণে গুরুত্বপূর্ণ।"),
    "pancreatic_cancer": ("Pancreatic cancer can involve the pancreas and nearby bile or digestive pathways; location, resectability, stage and pathology are important.", "অগ্ন্যাশয়ের ক্যান্সার pancreas ও কাছের bile/digestive pathway-তে প্রভাব ফেলতে পারে; অবস্থান, resectability, stage ও pathology গুরুত্বপূর্ণ।"),
    "liver_cancer": ("Liver cancer must be interpreted alongside liver function, hepatitis or cirrhosis history, lesion features, stage and pathology when indicated.", "লিভার ক্যান্সার বোঝার সময় liver function, hepatitis বা cirrhosis-এর ইতিহাস, lesion-এর বৈশিষ্ট্য, stage এবং প্রয়োজন হলে pathology দেখা হয়।"),
    "thyroid_cancer": ("Thyroid cancer includes several subtypes; ultrasound features, biopsy, thyroid function, spread and pathology help determine the next step.", "থাইরয়েড ক্যান্সারের কয়েকটি subtype আছে; ultrasound feature, biopsy, thyroid function, ছড়িয়ে পড়ার তথ্য ও pathology পরবর্তী পদক্ষেপে সাহায্য করে।"),
    "skin_cancer": ("Skin cancer includes different diseases such as basal-cell, squamous-cell and melanoma; lesion change, dermoscopy and biopsy are often important.", "ত্বকের ক্যান্সারের মধ্যে basal-cell, squamous-cell ও melanoma-এর মতো ধরন আছে; lesion-এর পরিবর্তন, dermoscopy ও biopsy গুরুত্বপূর্ণ হতে পারে।"),
    "brain_cancer": ("Brain tumors vary widely by cell type, location and grade; MRI findings alone do not establish the final diagnosis, and tissue or specialist review may be needed.", "মস্তিষ্কের tumour cell type, location ও grade অনুযায়ী অনেক ভিন্ন হতে পারে; শুধু MRI দিয়ে চূড়ান্ত diagnosis হয় না, tissue বা specialist review প্রয়োজন হতে পারে।"),
    "blood_cancer": ("Blood cancers include leukemia, lymphoma and myeloma; blood counts, cell markers, marrow or tissue studies and genetics may be relevant.", "রক্তের ক্যান্সারের মধ্যে leukemia, lymphoma ও myeloma আছে; blood count, cell marker, marrow বা tissue study এবং genetics গুরুত্বপূর্ণ হতে পারে।"),
}

def _topic_context(topic: CancerTopic) -> tuple[str, str]:
    return _TOPIC_CONTEXT.get(topic.key, (
        f"{topic.label} is a specific cancer topic. The meaningful explanation depends on its exact subtype, location, stage, biomarkers, symptoms and the evidence available.",
        f"{topic.label} একটি নির্দিষ্ট cancer topic। সঠিক ব্যাখ্যা subtype, location, stage, biomarker, উপসর্গ এবং পাওয়া evidence-এর ওপর নির্ভর করে।",
    ))

def _question_focus(question: str, language: str) -> str:
    text = question.casefold()
    if any(term in text for term in ("treatment", "therapy", "medicine", "চিকিৎসা", "ওষুধ")):
        return "The treatment discussion should be tailored to subtype, stage, biomarkers, goals, and overall health." if language == "en" else "চিকিৎসা নিয়ে আলোচনা subtype, stage, biomarker, treatment goal ও সামগ্রিক স্বাস্থ্যের ওপর নির্ভর করে।"
    if any(term in text for term in ("test", "diagnos", "biopsy", "পরীক্ষা", "নির্ণয়", "বায়োপসি")):
        return "Testing usually combines clinical assessment with the tests appropriate for the suspected site; pathology may be needed for confirmation." if language == "en" else "পরীক্ষায় clinical assessment-এর সঙ্গে সংশ্লিষ্ট অঙ্গের উপযুক্ত test লাগে; নিশ্চিতকরণের জন্য pathology প্রয়োজন হতে পারে।"
    if any(term in text for term in ("symptom", "sign", "লক্ষণ", "উপসর্গ")):
        return "Symptoms vary by site and stage, and common symptoms can have many non-cancer causes." if language == "en" else "উপসর্গ site ও stage অনুযায়ী বদলায়, আর সাধারণ উপসর্গের অনেক non-cancer কারণও থাকতে পারে।"
    if any(term in text for term in ("prevent", "risk", "screen", "প্রতিরোধ", "ঝুঁকি", "স্ক্রিন")):
        return "Risk reduction and screening depend on age, family history, exposures, and local clinical guidance." if language == "en" else "ঝুঁকি কমানো ও screening বয়স, পারিবারিক ইতিহাস, exposure এবং স্থানীয় clinical guidance-এর ওপর নির্ভর করে।"
    return "Ask about the exact subtype, stage, tests, treatment goals, and what evidence supports each conclusion." if language == "en" else "সঠিক subtype, stage, পরীক্ষা, চিকিৎসার লক্ষ্য এবং কোন evidence থেকে সিদ্ধান্তটি এসেছে—এসব জানতে পারেন।"


def _clean_text(value: str, *, max_length: int) -> str:
    cleaned = _CONTROL_CHARS.sub(" ", str(value or "")).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or len(cleaned) > max_length:
        raise ValueError("Cancer topic and question must be non-empty and bounded")
    return cleaned


def normalize_topic(value: str) -> CancerTopic:
    cleaned = _clean_text(value, max_length=120)
    lookup = cleaned.casefold()
    if lookup in _ALIASES:
        key, label = _ALIASES[lookup]
        return CancerTopic(key=key, label=label, registry_key=key)
    key = re.sub(r"[^a-z0-9]+", "_", lookup).strip("_") or "cancer"
    label = cleaned[0].upper() + cleaned[1:]
    registry_key = key if get_cancer_capability(key) else None
    return CancerTopic(key=key, label=label, registry_key=registry_key)


def _sources(topic: CancerTopic) -> list[InformationSource]:
    query = quote_plus(topic.label)
    return [
        InformationSource(
            title="National Cancer Institute — Cancer Types",
            url="https://www.cancer.gov/types",
            source_type="government_reference",
        ),
        InformationSource(
            title="National Cancer Institute — About Cancer",
            url=f"https://www.cancer.gov/search/results?swKeyword={query}",
            source_type="government_reference",
        ),
        InformationSource(
            title="MedlinePlus — Cancer",
            url="https://medlineplus.gov/cancer.html",
            source_type="government_reference",
        ),
        InformationSource(
            title="World Health Organization — Cancer",
            url="https://www.who.int/news-room/fact-sheets/detail/cancer",
            source_type="international_reference",
        ),
    ]


def _sources(topic: CancerTopic) -> list[InformationSource]:
    query = quote_plus(topic.label)
    return [
        InformationSource(
            title="National Cancer Institute - Cancer Types",
            url="https://www.cancer.gov/types",
            source_type="government_reference",
        ),
        InformationSource(
            title="National Cancer Institute - About Cancer",
            url=f"https://www.cancer.gov/search/results?swKeyword={query}",
            source_type="government_reference",
        ),
        InformationSource(
            title="MedlinePlus - Cancer",
            url="https://medlineplus.gov/cancer.html",
            source_type="government_reference",
        ),
        InformationSource(
            title="World Health Organization - Cancer",
            url="https://www.who.int/news-room/fact-sheets/detail/cancer",
            source_type="international_reference",
        ),
    ]


def _section(title: str, content: str, bullets: list[str] | None = None) -> InformationSection:
    return InformationSection(title=title, content=content, bullets=bullets or [])


class CancerInformationService:
    """Answer broad cancer questions with a safe fallback or configured LLM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _language(request: CancerInformationRequest) -> str:
        if request.language in {"bn", "en"}:
            return request.language
        return "bn" if _BENGALI.search(request.question) else "en"

    def answer(self, request: CancerInformationRequest) -> CancerInformationResponse:
        topic = normalize_topic(request.cancer)
        question = _clean_text(request.question, max_length=2400)
        language = self._language(request)
        model_response = self._configured_model_answer(topic, question, language)
        if model_response is not None:
            return model_response
        return self._fallback_answer(topic, question, language)

    def _configured_model_answer(
        self, topic: CancerTopic, question: str, language: str
    ) -> CancerInformationResponse | None:
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        api_key = os.getenv("ONCOAEGIS_CANCER_AI_API_KEY") or gemini_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = (
            os.getenv("ONCOAEGIS_CANCER_AI_BASE_URL")
            or os.getenv("GEMINI_BASE_URL")
            or (
                "https://generativelanguage.googleapis.com/v1beta/openai"
                if gemini_key
                else "https://api.openai.com/v1"
            )
        ).rstrip("/")
        configured_model = os.getenv("ONCOAEGIS_CANCER_AI_MODEL") or os.getenv("GEMINI_MODEL")
        model = configured_model or (
            "gemini-2.5-flash"
            if "generativelanguage.googleapis.com" in base_url
            else "gpt-4o-mini"
        )
        prompt_language = "Bangla" if language == "bn" else "English"
        system = (
            "You are Onco Aegis AI, a careful cancer-information educator. "
            "Answer in the requested language. Provide general education only; "
            "never diagnose, stage, rule out cancer, prescribe a drug or dose, "
            "or tell a person to start/stop treatment. Explain that pathology, "
            "clinical history, examination and appropriate testing may be needed. "
            "Separate common possibilities from warning signs, mention uncertainty, "
            "and advise urgent local care for emergency symptoms. Do not request "
            "names, addresses, medical record numbers or other identifying data. "
            "Return strict JSON with keys: answer, sections (array of objects with "
            "title, content, bullets), follow_up_questions (array), urgent_guidance."
        )
        user = json.dumps(
            {
                "cancer_topic": topic.label,
                "question": question,
                "language": prompt_language,
                "known_specialist_registry_match": bool(topic.registry_key),
                "reference_urls": [source.url for source in _sources(topic)],
            },
            ensure_ascii=False,
        )
        body_payload = {
            "model": model,
            "temperature": 0.15,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = json.dumps(body_payload).encode("utf-8")
        request = Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            if content.strip().startswith("```"):
                content = content.strip().split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            sections = [InformationSection.model_validate(item) for item in parsed.get("sections", [])]
            if not isinstance(parsed.get("answer"), str) or not sections:
                return None
            return CancerInformationResponse(
                request_id=f"info-{uuid.uuid4().hex}",
                cancer=topic.label,
                answer=parsed["answer"],
                sections=sections,
                follow_up_questions=[str(item) for item in parsed.get("follow_up_questions", [])][:6],
                urgent_guidance=str(parsed.get("urgent_guidance", "Seek urgent local care for severe or rapidly worsening symptoms.")),
                sources=_sources(topic),
                mode="configured_ai",
                specialist_registry_match=bool(topic.registry_key),
                diagnostic_conclusion=False,
                expert_review_required=True,
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            UnicodeError,
            KeyError,
            IndexError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            # A provider outage must never turn into an invented medical answer.
            return None

    def _bangla_fallback(
        self,
        topic: CancerTopic,
        question: str,
        personal: bool,
        urgent: bool,
    ) -> CancerInformationResponse:
        _, topic_bn = _topic_context(topic)
        answer = (
            f"{topic.label} সম্পর্কে সাধারণভাবে বললে, এটি একটি নির্দিষ্ট ধরনের ক্যান্সার বা রোগগোষ্ঠী। "
            "তবে subtype, stage, biomarker, বয়স, উপসর্গ এবং সামগ্রিক স্বাস্থ্য অনুযায়ী তথ্যের অর্থ বদলাতে পারে। "
            "এই উত্তরটি শিক্ষামূলক; এটি রোগ নির্ণয় নয়।"
        )
        answer = topic_bn + " " + _question_focus(question, "bn") + " এটি সাধারণ তথ্য; ব্যক্তিগত diagnosis নয়। " + answer
        if personal:
            answer += " ব্যক্তিগত রিপোর্ট বা উপসর্গের নির্ভরযোগ্য ব্যাখ্যার জন্য চিকিৎসকের পরীক্ষা ও সংশ্লিষ্ট রিপোর্ট প্রয়োজন।"
        if urgent:
            answer += " আপনার প্রশ্নে জরুরি উপসর্গের ইঙ্গিত থাকায় সাধারণ তথ্যের জন্য চিকিৎসা নিতে দেরি করবেন না।"

        sections = [
            _section(
                "এটি কী",
                "ক্যান্সার হলো এমন একদল রোগ যেখানে কিছু কোষের বৃদ্ধি ও বিভাজন নিয়ন্ত্রণের বাইরে চলে যেতে পারে এবং কখনও আশপাশের টিস্যুতে ছড়িয়ে পড়তে পারে। নির্দিষ্ট cancer type ও subtype জানা না থাকলে খুব নির্দিষ্ট সিদ্ধান্ত দেওয়া নিরাপদ নয়।",
            ),
            _section(
                "সম্ভাব্য উপসর্গ",
                "উপসর্গ ক্যান্সারের ধরন ও অবস্থানের ওপর নির্ভর করে। অনেক উপসর্গের ক্যান্সার ছাড়া অন্য কারণও থাকতে পারে; শুধু উপসর্গ দেখে ক্যান্সার নিশ্চিত বা বাদ দেওয়া যায় না।",
                [
                    "দীর্ঘদিন থাকা গাঁট বা ফোলা",
                    "কারণ ছাড়া ওজন কমা বা দীর্ঘস্থায়ী দুর্বলতা",
                    "দীর্ঘস্থায়ী ব্যথা, রক্তপাত, বা পায়খানা/প্রস্রাবের অভ্যাসে পরিবর্তন",
                    "দীর্ঘস্থায়ী কাশি, শ্বাসকষ্ট, বা গিলতে সমস্যা",
                ],
            ),
            _section(
                "কীভাবে পরীক্ষা করা হয়",
                "চিকিৎসক রোগের ইতিহাস ও শারীরিক পরীক্ষা দেখে blood test, imaging, endoscopy বা অন্য পরীক্ষা বিবেচনা করতে পারেন। অনেক ক্ষেত্রে tissue diagnosis নিশ্চিত করতে biopsy ও pathology review প্রয়োজন হতে পারে; staging আলাদা ধাপ।",
                [
                    "উপসর্গ ও risk factor পর্যালোচনা",
                    "প্রয়োজন অনুযায়ী imaging বা laboratory evaluation",
                    "সন্দেহজনক tissue থাকলে pathology review",
                    "ধরন, stage, biomarker ও চিকিৎসার লক্ষ্য অনুযায়ী পরিকল্পনা",
                ],
            ),
            _section(
                "চিকিৎসা সম্পর্কে সাধারণ ধারণা",
                "চিকিৎসা cancer type, stage, biomarker, চিকিৎসার লক্ষ্য এবং সামগ্রিক স্বাস্থ্যের ওপর নির্ভর করে। surgery, radiation, chemotherapy, hormone therapy, targeted therapy বা immunotherapy বিভিন্নভাবে বিবেচিত হতে পারে। নিজের সিদ্ধান্তে কোনো ওষুধ শুরু বা বন্ধ করবেন না।",
            ),
            _section(
                "চিকিৎসককে যে প্রশ্ন করতে পারেন",
                "নির্দিষ্ট type/subtype, stage, প্রয়োজনীয় পরীক্ষা, চিকিৎসার লক্ষ্য, সম্ভাব্য উপকার ও ঝুঁকি, side-effect support, second opinion এবং clinical-trial option সম্পর্কে জিজ্ঞেস করতে পারেন।",
            ),
        ]
        urgent_guidance = (
            "গুরুতর শ্বাসকষ্ট, অজ্ঞান হওয়া বা confusion, প্রচুর রক্তপাত, নিয়ন্ত্রণহীন ব্যথা, "
            "অথবা cancer treatment চলাকালে উচ্চ জ্বর হলে এখনই স্থানীয় জরুরি সেবা নিন।"
        )
        return CancerInformationResponse(
            request_id=f"info-{uuid.uuid4().hex}",
            cancer=topic.label,
            answer=answer,
            sections=sections,
            follow_up_questions=[
                "আপনি কি উপসর্গ, পরীক্ষা, চিকিৎসা, prevention, নাকি clinical trial সম্পর্কে জানতে চান?",
                "আপনি কোন দেশ বা health system অনুযায়ী তথ্য চান?",
                "আপনি কি নির্দিষ্ট cancer type বা subtype বোঝাচ্ছেন?",
            ],
            urgent_guidance=urgent_guidance,
            sources=_sources(topic),
            mode="safe_fallback",
            specialist_registry_match=bool(topic.registry_key),
            diagnostic_conclusion=False,
            expert_review_required=True,
        )

    def _fallback_answer(
        self, topic: CancerTopic, question: str, language: str
    ) -> CancerInformationResponse:
        personal = bool(_PERSONAL_OR_DIAGNOSTIC.search(question))
        urgent = bool(_URGENT_TERMS.search(question))
        if language == "bn":
            return self._bangla_fallback(topic, question, personal, urgent)
        if language == "bn":
            answer = (
                f"{topic.label} সম্পর্কে সাধারণভাবে বললে, এটি একটি নির্দিষ্ট রোগের নাম/ধরন—"
                "কিন্তু একই নামের মধ্যেও subtype, stage, biomarkers, বয়স, উপসর্গ ও সামগ্রিক স্বাস্থ্য অনুযায়ী "
                "তথ্য ও চিকিৎসার পরিকল্পনা বদলাতে পারে। এই উত্তর শিক্ষামূলক; এটি diagnosis নয়।"
            )
            if personal:
                answer += " আপনার রিপোর্ট বা উপসর্গ থেকে নিশ্চিত সিদ্ধান্ত দিতে হলে চিকিৎসকের পরীক্ষা ও প্রাসঙ্গিক রিপোর্ট প্রয়োজন।"
            sections = [
                _section("এটি কী", "ক্যান্সার হলো কোষের নিয়ন্ত্রণহীন বৃদ্ধি ও ছড়িয়ে পড়ার সম্ভাবনাসহ একটি জটিল রোগগোষ্ঠী। নির্দিষ্ট cancer type ও subtype না জানলে সাধারণ তথ্যের বেশি বলা নিরাপদ নয়."),
                _section("সম্ভাব্য লক্ষণ", "লক্ষণ cancer-এর ধরন ও অবস্থানের ওপর নির্ভর করে। অনেক সাধারণ লক্ষণের অন্য কারণও থাকতে পারে; শুধু লক্ষণ দেখে ক্যান্সার নিশ্চিত করা যায় না.", ["অস্বাভাবিক বা দীর্ঘস্থায়ী গাঁট/ফোলা", "অকারণ ওজন কমা বা দীর্ঘস্থায়ী দুর্বলতা", "দীর্ঘদিনের ব্যথা, রক্তপাত বা bowel/urinary habit-এর পরিবর্তন", "দীর্ঘস্থায়ী কাশি, শ্বাসকষ্ট বা গিলতে সমস্যা"]),
                _section("কীভাবে পরীক্ষা করা হয়", "চিকিৎসক history ও physical examination-এর সঙ্গে blood test, imaging, endoscopy বা অন্য পরীক্ষা বিবেচনা করতে পারেন। অনেক ক্ষেত্রে diagnosis নিশ্চিত করতে biopsy/pathology গুরুত্বপূর্ণ; staging আলাদা ধাপ.", ["প্রথমে উপসর্গ ও risk factors পর্যালোচনা", "প্রয়োজন অনুযায়ী imaging বা laboratory evaluation", "সন্দেহজনক tissue থাকলে pathology review", "type ও stage অনুযায়ী treatment planning"]),
                _section("চিকিৎসা সম্পর্কে সাধারণ ধারণা", "চিকিৎসা cancer type, stage, biomarkers, treatment goal এবং ব্যক্তির স্বাস্থ্য অনুযায়ী নির্ধারিত হয়। surgery, radiation, chemotherapy, hormone therapy, targeted therapy বা immunotherapy—একটি বা একাধিক পদ্ধতি বিবেচিত হতে পারে; নিজে থেকে কোনো ওষুধ শুরু/বন্ধ করা উচিত নয়."),
                _section("চিকিৎসককে কী জিজ্ঞেস করবেন", "পরবর্তী appointment-এ type/subtype, stage, প্রয়োজনীয় পরীক্ষা, treatment-এর লক্ষ্য, side effects, second opinion এবং clinical-trial options সম্পর্কে জিজ্ঞেস করতে পারেন."),
            ]
            follow_up = ["আপনি কি সাধারণ লক্ষণ, পরীক্ষা, চিকিৎসা, নাকি prevention সম্পর্কে জানতে চান?", "কোন দেশের/health system-এর তথ্য দরকার?", "আপনি কি নির্দিষ্ট cancer type বা subtype বোঝাচ্ছেন?"]
            urgent_guidance = "যদি শ্বাস নিতে গুরুতর কষ্ট, অচেতনতা/confusion, প্রচুর রক্তপাত, uncontrolled pain, বা চিকিৎসা চলাকালে উচ্চ জ্বর থাকে, এখনই স্থানীয় জরুরি সেবা নিন।"
        else:
            topic_en, _ = _topic_context(topic)
            answer = (
                f"{topic_en} {_question_focus(question, 'en')} This is educational information, not a diagnosis."
            )
            if personal:
                answer += " A clinician needs the relevant history, examination, and reports to interpret an individual situation."
            sections = [
                _section("What it is", "Cancer is a broad group of diseases involving abnormal cell growth and possible spread. The safest explanation depends on the specific type and subtype."),
                _section("Possible symptoms", "Symptoms vary by cancer type and location, and many have non-cancer causes. Symptoms alone cannot confirm or exclude cancer.", ["A persistent lump or swelling", "Unexplained weight loss or persistent fatigue", "Persistent pain, bleeding, or bowel/urinary changes", "A long-lasting cough, breathlessness, or difficulty swallowing"]),
                _section("How doctors evaluate it", "A clinician may combine history and examination with blood tests, imaging, endoscopy, or other investigations. A biopsy and pathology review may be needed to confirm many cancers; staging is a separate step.", ["Review symptoms and risk factors", "Choose appropriate imaging or laboratory tests", "Obtain pathology when a tissue diagnosis is needed", "Plan care based on type, stage, biomarkers, and goals"]),
                _section("Treatment overview", "Treatment depends on cancer type, stage, biomarkers, treatment goals, and overall health. Surgery, radiation, chemotherapy, hormone therapy, targeted therapy, and immunotherapy may be considered in different combinations. Do not start or stop treatment without the treating team."),
                _section("Questions for the care team", "Ask about the exact type/subtype, stage, tests still needed, treatment goals, expected benefits and risks, side-effect support, second opinions, and relevant clinical-trial options."),
            ]
            follow_up = ["Would you like symptoms, testing, treatment, prevention, or clinical-trial information?", "Which country or health system should the information fit?", "Do you mean a specific cancer type or subtype?"]
            urgent_guidance = "Seek local emergency care for severe breathing difficulty, confusion or fainting, heavy bleeding, uncontrolled pain, or high fever during cancer treatment."
        if urgent:
            answer += " Because your question mentions a potentially urgent symptom, general information should not delay prompt medical assessment."
        return CancerInformationResponse(
            request_id=f"info-{uuid.uuid4().hex}",
            cancer=topic.label,
            answer=answer,
            sections=sections,
            follow_up_questions=follow_up,
            urgent_guidance=urgent_guidance,
            sources=_sources(topic),
            mode="safe_fallback",
            specialist_registry_match=bool(topic.registry_key),
            diagnostic_conclusion=False,
            expert_review_required=True,
        )


_service: CancerInformationService | None = None
_service_lock = threading.Lock()


def get_cancer_information_service() -> CancerInformationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = CancerInformationService()
    return _service
