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
    "adrenal": ("adrenal_cancer", "Adrenal cancer"),
    "adrenal cancer": ("adrenal_cancer", "Adrenal cancer"),
    "adrenocortical carcinoma": ("adrenal_cancer", "Adrenal cancer"),
    "adrenocortical carcinoma acc": ("adrenal_cancer", "Adrenal cancer"),
    "acc": ("adrenal_cancer", "Adrenal cancer"),
    "prostate": ("prostate_cancer", "Prostate cancer"),
    "prostate cancer": ("prostate_cancer", "Prostate cancer"),
    "prostate adenocarcinoma": ("prostate_cancer", "Prostate cancer"),
    "kidney cancer": ("kidney_cancer", "Kidney cancer"),
    "renal cell carcinoma": ("kidney_cancer", "Kidney cancer"),
    "rcc": ("kidney_cancer", "Kidney cancer"),
    "ovarian cancer": ("ovarian_cancer", "Ovarian cancer"),
    "ovarian carcinoma": ("ovarian_cancer", "Ovarian cancer"),
    "cervical cancer": ("cervical_cancer", "Cervical cancer"),
    "cervical carcinoma": ("cervical_cancer", "Cervical cancer"),
    "endometrial cancer": ("endometrial_cancer", "Endometrial cancer"),
    "uterine cancer": ("endometrial_cancer", "Endometrial cancer"),
    "stomach cancer": ("stomach_cancer", "Stomach cancer"),
    "gastric cancer": ("stomach_cancer", "Stomach cancer"),
    "esophageal cancer": ("esophageal_cancer", "Esophageal cancer"),
    "oesophageal cancer": ("esophageal_cancer", "Esophageal cancer"),
    "bladder cancer": ("bladder_cancer", "Bladder cancer"),
    "urothelial carcinoma": ("bladder_cancer", "Bladder cancer"),
    "melanoma": ("melanoma", "Melanoma"),
    "multiple myeloma": ("multiple_myeloma", "Multiple myeloma"),
    "myeloma": ("multiple_myeloma", "Multiple myeloma"),
    "hodgkin lymphoma": ("hodgkin_lymphoma", "Hodgkin lymphoma"),
    "non hodgkin lymphoma": ("non_hodgkin_lymphoma", "Non-Hodgkin lymphoma"),
    "non-hodgkin lymphoma": ("non_hodgkin_lymphoma", "Non-Hodgkin lymphoma"),
    "testicular cancer": ("testicular_cancer", "Testicular cancer"),
    "gallbladder cancer": ("gallbladder_cancer", "Gallbladder cancer"),
    "cholangiocarcinoma": ("bile_duct_cancer", "Bile duct cancer"),
    "bile duct cancer": ("bile_duct_cancer", "Bile duct cancer"),
    "sarcoma": ("sarcoma", "Sarcoma"),
    "oral cancer": ("oral_cancer", "Oral cancer"),
    "mouth cancer": ("oral_cancer", "Oral cancer"),
    "head and neck cancer": ("head_neck_cancer", "Head and neck cancer"),
    "oropharyngeal cancer": ("head_neck_cancer", "Head and neck cancer"),
    "nasopharyngeal cancer": ("head_neck_cancer", "Head and neck cancer"),
    "anal cancer": ("anal_cancer", "Anal cancer"),
    "small intestine cancer": ("small_intestine_cancer", "Small intestine cancer"),
    "small bowel cancer": ("small_intestine_cancer", "Small intestine cancer"),
    "appendix cancer": ("appendix_cancer", "Appendix cancer"),
    "neuroendocrine tumor": ("neuroendocrine_tumor", "Neuroendocrine tumor"),
    "bone cancer": ("bone_cancer", "Bone cancer"),
    "osteosarcoma": ("bone_cancer", "Bone cancer"),
    "ewing sarcoma": ("bone_cancer", "Bone cancer"),
    "soft tissue sarcoma": ("soft_tissue_sarcoma", "Soft tissue sarcoma"),
    "mesothelioma": ("mesothelioma", "Mesothelioma"),
    "thymic cancer": ("thymic_cancer", "Thymic cancer"),
    "thymoma": ("thymic_cancer", "Thymic cancer"),
    "salivary gland cancer": ("salivary_gland_cancer", "Salivary gland cancer"),
    "vulvar cancer": ("vulvar_cancer", "Vulvar cancer"),
    "vaginal cancer": ("vaginal_cancer", "Vaginal cancer"),
    "uterine sarcoma": ("uterine_sarcoma", "Uterine sarcoma"),
    "urethral cancer": ("urethral_cancer", "Urethral cancer"),
    "ureter cancer": ("upper_urinary_tract_cancer", "Upper urinary tract cancer"),
    "renal pelvis cancer": ("upper_urinary_tract_cancer", "Upper urinary tract cancer"),
    "glioblastoma": ("brain_cancer", "Brain cancer"),
    "spinal cord tumor": ("spinal_cord_tumor", "Spinal cord tumor"),
    "peritoneal cancer": ("peritoneal_cancer", "Peritoneal cancer"),
    "mesenteric cancer": ("peritoneal_cancer", "Peritoneal cancer"),
    "kaposi sarcoma": ("kaposi_sarcoma", "Kaposi sarcoma"),
    "burkitt lymphoma": ("burkitt_lymphoma", "Burkitt lymphoma"),
    "mantle cell lymphoma": ("mantle_cell_lymphoma", "Mantle cell lymphoma"),
    "mycosis fungoides": ("cutaneous_t_cell_lymphoma", "Cutaneous T-cell lymphoma"),
    "chronic lymphocytic leukemia": ("chronic_lymphocytic_leukemia", "Chronic lymphocytic leukemia"),
    "chronic myeloid leukemia": ("chronic_myeloid_leukemia", "Chronic myeloid leukemia"),
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

_TOPIC_CONTEXT.update({
    "adrenal_cancer": ("Adrenal cancer includes rare tumors such as adrenocortical carcinoma. Hormone production, tumor size, spread, imaging and pathology are important to specialist evaluation.", ""),
    "prostate_cancer": ("Prostate cancer commonly refers to prostate adenocarcinoma. PSA, examination, imaging, biopsy grade, stage and clinical risk group guide evaluation.", ""),
    "kidney_cancer": ("Kidney cancer often refers to renal-cell carcinoma. Kidney function, imaging pattern, tumor size, spread and pathology help guide evaluation.", ""),
    "ovarian_cancer": ("Ovarian cancer includes several epithelial and non-epithelial subtypes. Symptoms can be vague; pelvic imaging, tumor markers and tissue diagnosis may be used together.", ""),
    "cervical_cancer": ("Cervical cancer is often related to persistent high-risk HPV infection. Screening, examination, biopsy and stage assessment are central to evaluation.", ""),
    "endometrial_cancer": ("Endometrial cancer begins in the uterine lining. Abnormal uterine bleeding is an important symptom to assess, but many non-cancer causes are possible.", ""),
    "stomach_cancer": ("Stomach, or gastric, cancer requires assessment of symptoms, endoscopy and biopsy; subtype, location, stage and biomarker findings affect care discussions.", ""),
    "esophageal_cancer": ("Esophageal cancer affects the swallowing tube. Progressive swallowing difficulty, weight loss and reflux-like symptoms need clinical assessment because they have multiple possible causes.", ""),
    "bladder_cancer": ("Bladder cancer is often evaluated after blood in the urine or urinary symptoms. Urine testing, imaging and cystoscopy with pathology may be relevant.", ""),
    "melanoma": ("Melanoma is a skin cancer arising from pigment-producing cells. A changing or unusual lesion needs clinical and dermatology assessment; biopsy establishes diagnosis.", ""),
    "multiple_myeloma": ("Multiple myeloma is a plasma-cell cancer. Blood counts, kidney function, calcium, protein studies, imaging and marrow assessment may be relevant.", ""),
    "hodgkin_lymphoma": ("Hodgkin lymphoma is a lymphoma subtype involving lymphatic tissue. Persistent lymph-node swelling and systemic symptoms need medical assessment and tissue review.", ""),
    "non_hodgkin_lymphoma": ("Non-Hodgkin lymphoma is a broad group of lymphoid cancers with many subtypes. Tissue pathology, immunophenotyping and imaging help define the specific disease.", ""),
    "testicular_cancer": ("Testicular cancer often presents as a persistent testicular lump or enlargement. Examination, ultrasound, blood markers and specialist review are important.", ""),
    "gallbladder_cancer": ("Gallbladder cancer is uncommon and may be found during evaluation of biliary or abdominal symptoms. Imaging and pathology are needed for reliable interpretation.", ""),
    "bile_duct_cancer": ("Bile duct cancer, or cholangiocarcinoma, can affect bile drainage. Jaundice, itching or abdominal symptoms have many causes and require clinical evaluation.", ""),
    "sarcoma": ("Sarcoma is a diverse group of cancers of connective or supporting tissues. A growing deep or persistent mass needs imaging and specialist pathology review.", ""),
})


_OFFLINE_PROFILES = {
    "oral_cancer": {"overview": "Oral cancer can affect the lips, tongue, gums, cheek, floor of mouth, or other mouth tissues.", "symptoms": ["A mouth sore that does not heal", "A persistent lump, pain, numbness, bleeding, or red/white patch", "Difficulty chewing, swallowing, or moving the jaw"], "tests": "Mouth and neck examination, dental or ENT review, imaging when needed, and biopsy of a concerning lesion.", "prevention": "Avoid tobacco and betel-nut exposure, limit alcohol, use sun protection for the lips, and attend dental or medical review for persistent lesions."},
    "head_neck_cancer": {"overview": "Head and neck cancers arise in areas such as the throat, voice box, nose, sinuses, or nearby tissues; HPV and tobacco are relevant risks for some types.", "symptoms": ["Persistent hoarseness or sore throat", "A neck lump, swallowing difficulty, ear pain, or nasal blockage", "Unexplained weight loss or a mouth/throat lesion"], "tests": "ENT examination, endoscopy, imaging, HPV-related testing when relevant, and tissue biopsy.", "prevention": "Do not smoke or use smokeless tobacco, limit alcohol, consider HPV vaccination according to local guidance, and seek review for persistent symptoms."},
    "anal_cancer": {"overview": "Anal cancer develops in the anal canal or nearby skin and has several possible cell types.", "symptoms": ["Bleeding, pain, itching, or a lump near the anus", "Change in bowel habits or discharge"], "tests": "Examination, anoscopy or endoscopy, imaging for extent, and biopsy.", "prevention": "HPV vaccination, safer-sex practices, avoiding tobacco, and assessment of persistent anal symptoms may reduce risk or delay."},
    "small_intestine_cancer": {"overview": "Small-intestine cancers include adenocarcinoma, neuroendocrine tumors, lymphoma, and sarcoma, so subtype matters.", "symptoms": ["Abdominal pain, cramps, nausea, vomiting, bleeding, or unexplained weight loss", "Anemia or symptoms of bowel blockage"], "tests": "Blood tests, CT or MRI, endoscopy or capsule studies when appropriate, and biopsy.", "prevention": "There is no general screening test for most people; manage known bowel disease and inherited-risk conditions with specialist guidance."},
    "appendix_cancer": {"overview": "Appendix tumors include neuroendocrine tumors and mucinous or other neoplasms; their behavior varies widely.", "symptoms": ["Appendicitis-like pain, abdominal swelling, or bowel changes", "Many are found unexpectedly during surgery or imaging"], "tests": "Imaging, surgery findings, pathology, and sometimes tumor-marker or staging assessment.", "prevention": "No proven general prevention or population screening exists; follow specialist advice for inherited-risk syndromes."},
    "neuroendocrine_tumor": {"overview": "Neuroendocrine tumors can arise in many organs and may grow slowly or aggressively; hormone production is relevant for some tumors.", "symptoms": ["Symptoms depend on the organ and hormone production", "Flushing, diarrhea, wheezing, pain, or an incidental mass can occur"], "tests": "Organ-specific imaging, blood or urine markers when appropriate, pathology and receptor or biomarker testing.", "prevention": "Most have no established prevention; manage inherited syndromes and follow specialist surveillance when applicable."},
    "bone_cancer": {"overview": "Primary bone cancers include osteosarcoma, Ewing sarcoma, and other rare tumors; cancers that spread to bone are a separate situation.", "symptoms": ["Persistent localized bone pain or swelling", "A lump, reduced movement, or a fracture after minor injury"], "tests": "X-ray or MRI, staging scans, and carefully planned biopsy by a specialist team.", "prevention": "There is no proven general prevention; persistent bone pain or swelling should be assessed rather than self-diagnosed."},
    "soft_tissue_sarcoma": {"overview": "Soft-tissue sarcoma is a group of cancers arising in muscle, fat, blood vessels, nerves, or connective tissue.", "symptoms": ["A growing or deep lump, often initially painless", "Pain, pressure, or reduced function as a mass enlarges"], "tests": "MRI or CT followed by specialist-planned core biopsy and pathology review.", "prevention": "Most cases have no known preventable cause; avoid unnecessary radiation exposure and discuss inherited risk when family history suggests it."},
    "mesothelioma": {"overview": "Mesothelioma is a cancer of lining tissues, most often associated with the pleura; asbestos exposure is an important risk factor.", "symptoms": ["Shortness of breath, chest pain, persistent cough, or fluid around the lung", "Abdominal swelling or pain in peritoneal disease"], "tests": "Imaging, fluid or tissue sampling, pathology, and specialist staging.", "prevention": "Avoid occupational or environmental asbestos exposure and follow workplace protection rules; smoking cessation remains important for overall lung health."},
    "thymic_cancer": {"overview": "Thymic tumors arise in the thymus in the chest and include thymoma and thymic carcinoma.", "symptoms": ["Chest discomfort, cough, breathlessness, or difficulty swallowing", "Some people have no symptoms and are found incidentally"], "tests": "Chest imaging, specialist assessment, pathology, and evaluation for associated autoimmune conditions when indicated.", "prevention": "No proven general prevention exists; treatment and surveillance depend on tumor type and extent."},
    "salivary_gland_cancer": {"overview": "Salivary-gland cancers are uncommon and include multiple histologic subtypes with different behavior.", "symptoms": ["A persistent cheek, jaw, mouth, or neck lump", "Facial weakness, numbness, pain, or difficulty opening the mouth"], "tests": "Head and neck examination, ultrasound or MRI, needle sampling or biopsy, and staging imaging.", "prevention": "There is no established general prevention; avoid tobacco and seek assessment for a persistent salivary-area lump."},
    "vulvar_cancer": {"overview": "Vulvar cancer affects the external genital skin and has squamous and other subtypes.", "symptoms": ["Persistent itching, pain, bleeding, skin change, or a sore or lump", "A change that does not settle with routine care"], "tests": "Clinical examination and biopsy; imaging is used when staging requires it.", "prevention": "HPV vaccination, avoiding tobacco, and evaluation of persistent vulvar changes can reduce risk or delay."},
    "vaginal_cancer": {"overview": "Vaginal cancer is uncommon and may be related to HPV or other factors depending on subtype.", "symptoms": ["Unusual bleeding or discharge", "Pelvic pain, pain during sex, or a vaginal lump"], "tests": "Pelvic examination, biopsy, HPV-related assessment when relevant, and imaging for staging.", "prevention": "HPV vaccination, cervical screening where recommended, safer-sex practices, and tobacco avoidance are helpful risk-reduction measures."},
    "uterine_sarcoma": {"overview": "Uterine sarcomas are uncommon connective-tissue cancers of the uterus and differ from endometrial carcinoma.", "symptoms": ["Abnormal vaginal bleeding, pelvic pressure, pain, or a rapidly enlarging uterus", "Symptoms may overlap with benign conditions"], "tests": "Pelvic examination, ultrasound or MRI, surgery and pathology; specialist staging is important.", "prevention": "There is no reliable general prevention; discuss unusual bleeding or inherited risk with a clinician."},
    "urethral_cancer": {"overview": "Urethral cancer is rare and can arise from different cell types along the urethra.", "symptoms": ["Blood in urine, bleeding from the urethra, a lump, weak stream, or difficulty urinating", "Pain or recurrent urinary symptoms"], "tests": "Urine testing, examination, urethroscopy, imaging, and biopsy.", "prevention": "No proven general screening exists; avoid tobacco and seek assessment for persistent urinary bleeding or obstruction."},
    "upper_urinary_tract_cancer": {"overview": "Upper urinary tract cancer includes urothelial tumors of the renal pelvis or ureter.", "symptoms": ["Blood in the urine, flank pain, or urinary symptoms", "Some cases are detected during evaluation of hematuria"], "tests": "Urine studies, CT or MRI urography, endoscopic assessment, and pathology.", "prevention": "Avoid tobacco and relevant workplace chemical exposures; persistent blood in urine needs medical assessment."},
    "spinal_cord_tumor": {"overview": "Spinal cord and spinal-column tumors may be primary or metastatic and can affect nerves through pressure.", "symptoms": ["Persistent back pain, weakness, numbness, balance change, or bladder/bowel difficulty", "New or worsening neurological symptoms require prompt assessment"], "tests": "Neurological examination and MRI; tissue diagnosis may be needed depending on location and safety.", "prevention": "There is no general prevention; new weakness, loss of sensation, or bladder/bowel dysfunction needs urgent medical care."},
    "peritoneal_cancer": {"overview": "Peritoneal cancer affects the lining of the abdomen and may be primary or related to another organ cancer.", "symptoms": ["Abdominal swelling, discomfort, early fullness, bowel changes, or unexplained weight loss", "Fluid accumulation may occur"], "tests": "CT or MRI, fluid analysis when present, tumor markers in context, and tissue pathology.", "prevention": "There is no general population screening; inherited ovarian or colorectal risk may warrant genetic counselling."},
    "kaposi_sarcoma": {"overview": "Kaposi sarcoma is a vascular tumor associated with human herpesvirus 8 and immune-system context.", "symptoms": ["Persistent purple, red, or brown skin lesions", "Mouth, lymph-node, lung, or digestive symptoms in internal disease"], "tests": "Skin or tissue biopsy, examination for extent, and assessment of immune status.", "prevention": "HIV prevention and treatment, safer sex, and appropriate immune-care follow-up are important risk-reduction measures."},
    "burkitt_lymphoma": {"overview": "Burkitt lymphoma is a fast-growing B-cell non-Hodgkin lymphoma that needs urgent specialist evaluation.", "symptoms": ["Rapidly enlarging lymph nodes, abdominal swelling or pain, fever, night sweats, or weight loss", "Symptoms vary by disease location"], "tests": "Urgent blood tests, imaging, tissue biopsy, immunophenotyping, genetics, and sometimes marrow or spinal-fluid studies.", "prevention": "There is no reliable general prevention; prompt evaluation of rapidly progressive symptoms is important."},
    "mantle_cell_lymphoma": {"overview": "Mantle-cell lymphoma is a B-cell non-Hodgkin lymphoma with variable behavior and specialized pathology markers.", "symptoms": ["Painless lymph-node swelling, fatigue, fever, night sweats, weight loss, or abdominal fullness", "Some people have few symptoms at first"], "tests": "Blood tests, imaging, lymph-node or tissue biopsy, immunophenotyping and genetic testing.", "prevention": "No proven general prevention exists; follow-up is individualized by a hematology team."},
    "cutaneous_t_cell_lymphoma": {"overview": "Cutaneous T-cell lymphomas primarily affect the skin and include mycosis fungoides and Sézary syndrome.", "symptoms": ["Persistent itchy patches, plaques, or tumors", "Widespread redness, lymph-node swelling, or blood involvement in advanced disease"], "tests": "Dermatology examination, repeated skin biopsies when needed, blood tests, and staging studies.", "prevention": "No established prevention exists; persistent or changing rashes should be assessed rather than treated indefinitely without review."},
    "chronic_lymphocytic_leukemia": {"overview": "Chronic lymphocytic leukemia is a blood and bone-marrow cancer involving abnormal lymphocytes; some people are monitored before treatment is needed.", "symptoms": ["Fatigue, recurrent infections, swollen lymph nodes, night sweats, weight loss, or enlarged spleen", "It may be found on a routine blood count"], "tests": "Complete blood count, flow cytometry, examination, and selected genetic or molecular tests.", "prevention": "No proven general prevention exists; infection prevention and regular specialist monitoring are important."},
    "chronic_myeloid_leukemia": {"overview": "Chronic myeloid leukemia is a myeloid blood cancer commonly defined by the BCR-ABL1 genetic change.", "symptoms": ["Fatigue, night sweats, weight loss, abdominal fullness, or bone discomfort", "It may be found through an abnormal blood count"], "tests": "Blood count, blood or marrow examination, and molecular testing for BCR-ABL1.", "prevention": "No proven general prevention exists; treatment response is monitored with regular molecular testing."},
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
        """Answer without allowing an optional provider to break the API.

        The configured model is an enhancement only.  A provider outage,
        incompatible response, or a deployment-time dependency mismatch must
        still return the local educational answer instead of HTTP 500.
        """
        topic = normalize_topic(request.cancer)
        question = _clean_text(request.question, max_length=2400)
        language = self._language(request)
        try:
            model_response = self._configured_model_answer(topic, question, language)
            if model_response is not None:
                return model_response
        except Exception:
            # Never expose provider/response-parser failures to the client.
            pass
        try:
            return self._fallback_answer(topic, question, language)
        except Exception:
            # Last-resort response for a partially configured production image.
            return self._emergency_fallback(topic, language)

    @staticmethod
    def _emergency_fallback(topic: CancerTopic, language: str) -> CancerInformationResponse:
        if language == "bn":
            answer = (
                f"{topic.label} সম্পর্কে এটি সাধারণ শিক্ষামূলক তথ্য; এটি diagnosis নয়। "
                "ব্যক্তিগত রিপোর্ট, উপসর্গ ও চিকিৎসার সিদ্ধান্ত qualified clinician-এর সঙ্গে আলোচনা করুন।"
            )
            title = "নিরাপদ তথ্য"
            content = "লক্ষণ দিয়ে cancer নিশ্চিত বা বাদ দেওয়া যায় না; প্রয়োজন হলে clinician পরীক্ষা, imaging ও pathology review করতে পারেন।"
        else:
            answer = (
                f"{topic.label} information is educational only and is not a diagnosis. "
                "Discuss personal reports, symptoms, and treatment decisions with a qualified clinician."
            )
            title = "Safe information"
            content = "Symptoms alone cannot confirm or exclude cancer; a clinician may use examination, imaging, laboratory testing, and pathology when appropriate."
        return CancerInformationResponse(
            request_id=f"info-{uuid.uuid4().hex}",
            cancer=topic.label,
            answer=answer,
            sections=[_section(title, content)],
            follow_up_questions=["Would you like symptoms, testing, treatment, or prevention information?"],
            urgent_guidance="Seek local emergency care for severe breathing difficulty, confusion or fainting, heavy bleeding, uncontrolled pain, or high fever during cancer treatment.",
            sources=_sources(topic),
            mode="safe_fallback",
            specialist_registry_match=bool(topic.registry_key),
            diagnostic_conclusion=False,
            expert_review_required=True,
        )

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

    def _offline_profile_answer(
        self, topic: CancerTopic, question: str, language: str
    ) -> CancerInformationResponse:
        profile = _OFFLINE_PROFILES[topic.key]
        answer = (
            f"{profile['overview']} {_question_focus(question, 'en')} "
            "This is educational information, not a diagnosis or a guarantee of cure."
        )
        if _PERSONAL_OR_DIAGNOSTIC.search(question):
            answer += " A clinician must interpret personal symptoms, reports, examination findings, and pathology."
        sections = [
            _section("What it is", profile["overview"]),
            _section("Possible symptoms", "These symptoms can have many non-cancer causes; symptoms alone cannot confirm or exclude cancer.", profile["symptoms"]),
            _section("How doctors evaluate it", profile["tests"], ["Clinical history and examination", "Imaging or laboratory tests selected for the suspected site", "Biopsy and pathology when tissue confirmation is needed", "Stage and biomarker assessment when relevant"]),
            _section("Prevention and risk reduction", profile["prevention"], ["Follow age- and risk-appropriate screening guidance", "Avoid tobacco and limit alcohol where relevant", "Discuss family history, inherited risk, vaccination, and occupational exposures with a clinician"]),
            _section("Treatment overview", "Treatment depends on subtype, stage, biomarkers, overall health, and goals. Surgery, radiation, chemotherapy, hormone therapy, targeted therapy, immunotherapy, or active monitoring may be considered in different situations. Do not start or stop treatment without the treating team."),
            _section("Questions for the care team", "Ask about the exact subtype, stage, tests still needed, treatment goal, expected benefits and risks, side-effect support, second opinion, and relevant clinical trials."),
        ]
        urgent_guidance = "Seek local emergency care for severe breathing difficulty, confusion or fainting, heavy bleeding, uncontrolled pain, or high fever during cancer treatment."
        return CancerInformationResponse(
            request_id=f"info-{uuid.uuid4().hex}",
            cancer=topic.label,
            answer=answer,
            sections=sections,
            follow_up_questions=["Would you like symptoms, testing, treatment, prevention, or risk information?", "Do you mean a specific subtype or stage?", "Would you like help preparing questions for a clinician?"],
            urgent_guidance=urgent_guidance,
            sources=_sources(topic),
            mode="offline_knowledge_base",
            specialist_registry_match=bool(topic.registry_key),
            diagnostic_conclusion=False,
            expert_review_required=True,
        )

    def _fallback_answer(
        self, topic: CancerTopic, question: str, language: str
    ) -> CancerInformationResponse:
        personal = bool(_PERSONAL_OR_DIAGNOSTIC.search(question))
        urgent = bool(_URGENT_TERMS.search(question))
        if topic.key in _OFFLINE_PROFILES and language == "en":
            return self._offline_profile_answer(topic, question, language)
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
