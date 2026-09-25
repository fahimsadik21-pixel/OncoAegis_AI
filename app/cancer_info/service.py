"""Safe, source-aware cancer education with an offline knowledge base.

The local content is intentionally educational and disease-specific.  It is a
reliable fallback when an optional OpenAI-compatible provider is unavailable;
it must never act as a diagnosis, staging tool, or treatment prescription.
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
    r"\b(do i have|is this cancer|what stage|my scan|my report|diagnos)\b|"
    r"আমার|আমাকে|স্টেজ|ক্যান্সার হয়েছে|ক্যান্সার হয়েছে",
    re.IGNORECASE,
)
_URGENT_TERMS = re.compile(
    r"\b(emergency|urgent|severe bleeding|can't breathe|cannot breathe|"
    r"confusion|fainting|uncontrolled pain|high fever|chemotherapy fever)\b|"
    r"জরুরি|শ্বাসকষ্ট|অতিরিক্ত রক্তপাত|অজ্ঞান|খুব বেশি ব্যথা|জ্বর",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CancerTopic:
    key: str
    label: str
    registry_key: str | None


def _aliases(*items: tuple[str, str]) -> dict[str, tuple[str, str]]:
    return {alias: (key, label) for alias, key, label in items}


_ALIASES = _aliases(
    ("lung", "lung_cancer", "Lung cancer"), ("lung cancer", "lung_cancer", "Lung cancer"),
    ("breast", "breast_cancer", "Breast cancer"), ("breast cancer", "breast_cancer", "Breast cancer"),
    ("brain", "brain_cancer", "Brain cancer"), ("brain tumor", "brain_cancer", "Brain cancer"), ("glioblastoma", "brain_cancer", "Brain cancer"),
    ("liver", "liver_cancer", "Liver cancer"), ("liver cancer", "liver_cancer", "Liver cancer"), ("hepatocellular carcinoma", "liver_cancer", "Liver cancer"), ("hcc", "liver_cancer", "Liver cancer"),
    ("pancreas", "pancreatic_cancer", "Pancreatic cancer"), ("pancreatic cancer", "pancreatic_cancer", "Pancreatic cancer"),
    ("colon", "colon_cancer", "Colon cancer"), ("colon cancer", "colon_cancer", "Colon cancer"),
    ("colorectal", "colorectal_cancer", "Colorectal cancer"), ("colorectal cancer", "colorectal_cancer", "Colorectal cancer"), ("rectal cancer", "colorectal_cancer", "Colorectal cancer"),
    ("skin", "skin_cancer", "Skin cancer"), ("skin cancer", "skin_cancer", "Skin cancer"),
    ("thyroid", "thyroid_cancer", "Thyroid cancer"), ("thyroid cancer", "thyroid_cancer", "Thyroid cancer"),
    ("blood cancer", "blood_cancer", "Blood cancer"), ("leukemia", "blood_cancer", "Blood cancer"), ("leukaemia", "blood_cancer", "Blood cancer"),
    ("all", "acute_lymphoblastic_leukemia", "Acute lymphoblastic leukemia"), ("acute lymphoblastic leukemia", "acute_lymphoblastic_leukemia", "Acute lymphoblastic leukemia"),
    ("aml", "acute_myeloid_leukemia", "Acute myeloid leukemia"), ("acute myeloid leukemia", "acute_myeloid_leukemia", "Acute myeloid leukemia"),
    ("adrenal", "adrenal_cancer", "Adrenal cancer"), ("adrenal cancer", "adrenal_cancer", "Adrenal cancer"), ("adrenocortical carcinoma", "adrenal_cancer", "Adrenal cancer"), ("adrenocortical carcinoma acc", "adrenal_cancer", "Adrenal cancer"), ("acc", "adrenal_cancer", "Adrenal cancer"),
    ("prostate", "prostate_cancer", "Prostate cancer"), ("prostate cancer", "prostate_cancer", "Prostate cancer"), ("prostate adenocarcinoma", "prostate_cancer", "Prostate cancer"),
    ("kidney cancer", "kidney_cancer", "Kidney cancer"), ("renal cell carcinoma", "kidney_cancer", "Kidney cancer"), ("rcc", "kidney_cancer", "Kidney cancer"),
    ("ovarian cancer", "ovarian_cancer", "Ovarian cancer"), ("ovarian carcinoma", "ovarian_cancer", "Ovarian cancer"),
    ("cervical cancer", "cervical_cancer", "Cervical cancer"), ("cervical carcinoma", "cervical_cancer", "Cervical cancer"),
    ("endometrial cancer", "endometrial_cancer", "Endometrial cancer"), ("uterine cancer", "endometrial_cancer", "Endometrial cancer"),
    ("stomach cancer", "stomach_cancer", "Stomach cancer"), ("gastric cancer", "stomach_cancer", "Stomach cancer"),
    ("esophageal cancer", "esophageal_cancer", "Esophageal cancer"), ("oesophageal cancer", "esophageal_cancer", "Esophageal cancer"),
    ("bladder cancer", "bladder_cancer", "Bladder cancer"), ("urothelial carcinoma", "bladder_cancer", "Bladder cancer"),
    ("melanoma", "melanoma", "Melanoma"), ("multiple myeloma", "multiple_myeloma", "Multiple myeloma"), ("myeloma", "multiple_myeloma", "Multiple myeloma"),
    ("hodgkin lymphoma", "hodgkin_lymphoma", "Hodgkin lymphoma"), ("non hodgkin lymphoma", "non_hodgkin_lymphoma", "Non-Hodgkin lymphoma"), ("non-hodgkin lymphoma", "non_hodgkin_lymphoma", "Non-Hodgkin lymphoma"),
    ("testicular cancer", "testicular_cancer", "Testicular cancer"), ("gallbladder cancer", "gallbladder_cancer", "Gallbladder cancer"),
    ("cholangiocarcinoma", "bile_duct_cancer", "Bile duct cancer"), ("bile duct cancer", "bile_duct_cancer", "Bile duct cancer"),
    ("sarcoma", "sarcoma", "Sarcoma"), ("oral cancer", "oral_cancer", "Oral cancer"), ("mouth cancer", "oral_cancer", "Oral cancer"),
    ("head and neck cancer", "head_neck_cancer", "Head and neck cancer"), ("oropharyngeal cancer", "head_neck_cancer", "Head and neck cancer"), ("nasopharyngeal cancer", "head_neck_cancer", "Head and neck cancer"),
    ("anal cancer", "anal_cancer", "Anal cancer"), ("small intestine cancer", "small_intestine_cancer", "Small intestine cancer"), ("small bowel cancer", "small_intestine_cancer", "Small intestine cancer"),
    ("appendix cancer", "appendix_cancer", "Appendix cancer"), ("neuroendocrine tumor", "neuroendocrine_tumor", "Neuroendocrine tumor"),
    ("bone cancer", "bone_cancer", "Bone cancer"), ("osteosarcoma", "bone_cancer", "Bone cancer"), ("ewing sarcoma", "bone_cancer", "Bone cancer"),
    ("soft tissue sarcoma", "soft_tissue_sarcoma", "Soft tissue sarcoma"), ("mesothelioma", "mesothelioma", "Mesothelioma"),
    ("thymic cancer", "thymic_cancer", "Thymic cancer"), ("thymoma", "thymic_cancer", "Thymic cancer"), ("salivary gland cancer", "salivary_gland_cancer", "Salivary gland cancer"),
    ("vulvar cancer", "vulvar_cancer", "Vulvar cancer"), ("vaginal cancer", "vaginal_cancer", "Vaginal cancer"), ("uterine sarcoma", "uterine_sarcoma", "Uterine sarcoma"),
    ("urethral cancer", "urethral_cancer", "Urethral cancer"), ("ureter cancer", "upper_urinary_tract_cancer", "Upper urinary tract cancer"), ("renal pelvis cancer", "upper_urinary_tract_cancer", "Upper urinary tract cancer"),
    ("spinal cord tumor", "spinal_cord_tumor", "Spinal cord tumor"), ("peritoneal cancer", "peritoneal_cancer", "Peritoneal cancer"), ("mesenteric cancer", "peritoneal_cancer", "Peritoneal cancer"),
    ("kaposi sarcoma", "kaposi_sarcoma", "Kaposi sarcoma"), ("burkitt lymphoma", "burkitt_lymphoma", "Burkitt lymphoma"), ("mantle cell lymphoma", "mantle_cell_lymphoma", "Mantle cell lymphoma"),
    ("mycosis fungoides", "cutaneous_t_cell_lymphoma", "Cutaneous T-cell lymphoma"), ("cutaneous t cell lymphoma", "cutaneous_t_cell_lymphoma", "Cutaneous T-cell lymphoma"),
    ("chronic lymphocytic leukemia", "chronic_lymphocytic_leukemia", "Chronic lymphocytic leukemia"), ("cll", "chronic_lymphocytic_leukemia", "Chronic lymphocytic leukemia"),
    ("chronic myeloid leukemia", "chronic_myeloid_leukemia", "Chronic myeloid leukemia"), ("cml", "chronic_myeloid_leukemia", "Chronic myeloid leukemia"),
)

_ALIASES.update({
    "স্তন ক্যান্সার": ("breast_cancer", "Breast cancer"),
    "ফুসফুসের ক্যান্সার": ("lung_cancer", "Lung cancer"),
    "মস্তিষ্কের ক্যান্সার": ("brain_cancer", "Brain cancer"),
    "লিভার ক্যান্সার": ("liver_cancer", "Liver cancer"),
    "অগ্ন্যাশয়ের ক্যান্সার": ("pancreatic_cancer", "Pancreatic cancer"),
    "কোলন ক্যান্সার": ("colon_cancer", "Colon cancer"),
    "ত্বকের ক্যান্সার": ("skin_cancer", "Skin cancer"),
    "থাইরয়েড ক্যান্সার": ("thyroid_cancer", "Thyroid cancer"),
    "রক্তের ক্যান্সার": ("blood_cancer", "Blood cancer"),
    "অ্যাড্রিনাল ক্যান্সার": ("adrenal_cancer", "Adrenal cancer"),
    "পাকস্থলীর ক্যান্সার": ("stomach_cancer", "Stomach cancer"),
    "প্রোস্টেট ক্যান্সার": ("prostate_cancer", "Prostate cancer"),
    "কিডনি ক্যান্সার": ("kidney_cancer", "Kidney cancer"),
    "ডিম্বাশয়ের ক্যান্সার": ("ovarian_cancer", "Ovarian cancer"),
    "জরায়ুমুখের ক্যান্সার": ("cervical_cancer", "Cervical cancer"),
})


def _profile(overview: str, symptoms: list[str], tests: str, treatment: str, prevention: str) -> dict[str, Any]:
    return {"overview": overview, "symptoms": symptoms, "tests": tests, "treatment": treatment, "prevention": prevention}


# Each entry is deliberately concise.  These are education anchors, not a
# substitute for country-specific clinical guidance or a personal assessment.
_OFFLINE_PROFILES: dict[str, dict[str, Any]] = {
    "breast_cancer": _profile("Breast cancer starts in breast tissue; receptor status, HER2 status and stage influence care.", ["A new breast or underarm lump or thickening", "A change in breast shape, skin, nipple, or unexplained bloody nipple discharge", "Persistent focal breast pain can need assessment, although it has many causes"], "Clinical breast examination, diagnostic mammography and ultrasound are common; MRI or biopsy may be used when appropriate.", "Depending on subtype and stage, care may combine surgery, radiation, chemotherapy, hormone therapy, HER2-targeted therapy, immunotherapy, or monitoring.", "Follow locally recommended screening, limit alcohol, maintain physical activity, discuss inherited risk, and seek review for a new persistent change."),
    "lung_cancer": _profile("Lung cancer includes non-small-cell and small-cell types; pathology, stage and molecular markers guide treatment discussions.", ["A cough that persists or changes", "Coughing blood, chest pain, worsening breathlessness, or repeated chest infections", "Unexplained weight loss, fatigue, or hoarseness"], "For people at risk, low-dose CT screening may be appropriate. Evaluation can include chest CT, PET/CT, bronchoscopy or needle biopsy, and molecular testing.", "Treatment may include surgery, radiation, chemotherapy, immunotherapy, and targeted medicines when a matching molecular change is present.", "Do not smoke or vape tobacco products, avoid second-hand smoke and occupational carcinogens, and ask about low-dose CT screening if eligible."),
    "brain_cancer": _profile("Brain tumors vary greatly by cell type, location and grade. An MRI image alone does not establish a final diagnosis.", ["A new seizure or a seizure pattern that changes", "Progressive weakness, speech, vision, balance, or personality changes", "Persistent headache with vomiting or other neurological change"], "Neurological examination and contrast MRI are central. Surgery or biopsy may be needed to identify tumor type and grade; pathology and molecular testing are important.", "Specialist care can involve observation, surgery, radiation, chemotherapy, targeted treatment, or supportive rehabilitation depending on the tumor.", "Most brain tumors do not have a proven preventable cause. Avoid unnecessary ionizing radiation and seek urgent assessment for new severe neurological symptoms."),
    "liver_cancer": _profile("Primary liver cancer, often hepatocellular carcinoma, is interpreted together with liver function, hepatitis, cirrhosis, imaging features and stage.", ["Upper-right abdominal discomfort or swelling", "Jaundice, abdominal fluid build-up, or itching", "Unexplained weight loss, reduced appetite, or fatigue"], "Liver-protocol CT or MRI, blood tests including liver function and AFP where appropriate, and assessment for hepatitis or cirrhosis are used. Biopsy is selected carefully.", "Options may include surveillance, ablation, surgery, transplant assessment, artery-directed treatment, radiation, immunotherapy, targeted medicines, or supportive care.", "Hepatitis B vaccination, treatment of viral hepatitis, limiting alcohol, maintaining a healthy weight, and surveillance for high-risk liver disease can reduce risk or find disease earlier."),
    "pancreatic_cancer": _profile("Pancreatic cancer can affect digestion and bile drainage; location, vessel involvement, stage and pathology determine the care pathway.", ["Jaundice with dark urine, pale stool, or itching", "Persistent upper abdominal or back pain", "Weight loss, appetite change, or newly difficult-to-control diabetes"], "Pancreas-protocol CT or MRI, endoscopic ultrasound with biopsy when needed, liver tests and selected markers such as CA 19-9 may be used.", "Care may include surgery for selected resectable tumors, chemotherapy, radiation in selected situations, biliary procedures, symptom support, and molecular testing.", "Avoid tobacco, limit alcohol, manage chronic pancreatitis and diabetes with a clinician, and discuss family history or genetic counselling when relevant."),
    "colon_cancer": _profile("Colon cancer arises in the large bowel. Location, pathology, stage and molecular findings affect treatment planning.", ["A persistent change in bowel habit", "Blood in stool or iron-deficiency anemia", "Abdominal discomfort, unexplained weight loss, or fatigue"], "Colonoscopy with biopsy is the key test; CT, blood counts and CEA may help with staging or monitoring after diagnosis.", "Treatment commonly uses surgery, with chemotherapy for selected stages; radiation is more often used for rectal than colon tumors. Targeted or immunotherapy may be relevant for some molecular profiles.", "Follow screening guidance, do not smoke, keep physically active, limit alcohol and processed meat, and discuss family history or inflammatory bowel disease."),
    "colorectal_cancer": _profile("Colorectal cancer includes colon and rectal cancers; exact location, pathology, stage and molecular profile matter.", ["A change in bowel habit, narrower stool, or persistent urgency", "Rectal bleeding or unexplained anemia", "Abdominal or pelvic pain, weight loss, or fatigue"], "Evaluation usually includes colonoscopy with biopsy, CT or MRI for staging, blood counts and CEA. Rectal cancer commonly needs pelvic MRI.", "Treatment can combine surgery, chemotherapy, radiation for many rectal cancers, and targeted or immunotherapy for selected molecular subtypes.", "Age- and risk-appropriate screening, exercise, healthy weight, limiting alcohol and processed meat, and discussing family history are important."),
    "skin_cancer": _profile("Skin cancer includes basal-cell, squamous-cell and melanoma; a photo alone cannot reliably diagnose the type.", ["A new, changing, asymmetric, bleeding, or non-healing skin lesion", "A mole with change in size, color, border, or symptoms", "A persistent scaly patch, sore, or lump"], "Dermatology examination, dermoscopy and biopsy are used for a concerning lesion. Imaging is used only when appropriate for the specific diagnosis.", "Treatment may include excision, Mohs surgery, topical or local treatments, radiation, immunotherapy, targeted therapy, or monitoring depending on type and extent.", "Use shade, protective clothing and broad-spectrum sunscreen; avoid tanning beds and arrange skin review for a changing lesion."),
    "thyroid_cancer": _profile("Thyroid cancer has several subtypes. Ultrasound features, fine-needle aspiration, pathology and spread guide care.", ["A neck lump or enlarging thyroid nodule", "Persistent hoarseness, trouble swallowing, or a pressure sensation", "Swollen neck lymph nodes; many thyroid cancers have no symptoms"], "Neck ultrasound and fine-needle aspiration are common. Thyroid function tests give context but do not diagnose cancer; pathology and selected molecular tests may follow.", "Management may include active surveillance for selected small tumors, surgery, radioactive iodine, thyroid hormone suppression, targeted treatment, or other specialist care.", "There is no routine prevention for most cases. Avoid unnecessary radiation exposure and seek review of a persistent or growing neck lump."),
    "blood_cancer": _profile("Blood cancers include leukemia, lymphoma and myeloma, which are different diseases requiring subtype-specific testing.", ["Persistent fatigue, recurrent infections, unusual bruising or bleeding", "Persistent swollen lymph nodes, fevers, drenching night sweats, or weight loss", "Bone pain, breathlessness, or abdominal fullness depending on the subtype"], "A complete blood count, blood film, flow cytometry, bone-marrow or tissue biopsy, imaging and genetic tests may be needed to define the subtype.", "Treatment is highly subtype-specific and can include observation, chemotherapy, immunotherapy, targeted therapy, radiation, cellular therapy, or stem-cell transplantation.", "Most blood cancers have no single preventable cause. Avoid tobacco, manage immune suppression with specialists, and discuss unusual family patterns with a clinician."),
    "acute_lymphoblastic_leukemia": _profile("Acute lymphoblastic leukemia (ALL) is a fast-growing cancer of immature lymphoid blood cells and needs urgent hematology assessment.", ["Fatigue, pallor, fever, infections, bruising, or bleeding", "Bone or joint pain, swollen nodes, or abdominal fullness", "Headache or neurological symptoms can occur in some situations"], "CBC, blood film, bone-marrow testing, flow cytometry, chromosome/genetic testing and lumbar puncture are used to define ALL and guide risk assessment.", "Care is delivered by hematology teams and commonly includes multi-agent chemotherapy, targeted or immunotherapy in selected cases, central nervous system treatment, and sometimes transplant.", "There is no proven general prevention. New fever, bleeding, severe weakness, or rapidly worsening symptoms require prompt medical assessment."),
    "acute_myeloid_leukemia": _profile("Acute myeloid leukemia (AML) is a fast-growing myeloid blood cancer; genetic findings influence treatment.", ["Fatigue, infections or persistent fever", "Easy bruising, bleeding, tiny red spots, or pallor", "Shortness of breath, bone discomfort, or gum swelling in some cases"], "CBC, blood film, bone-marrow examination, flow cytometry and chromosome/mutation testing establish subtype and risk group.", "Treatment may include intensive or lower-intensity chemotherapy, targeted medicines, supportive transfusions, infection management and stem-cell transplant for selected people.", "There is no reliable general prevention. People with fever during chemotherapy or significant bleeding need urgent local medical care."),
    "adrenal_cancer": _profile("Adrenal cancer includes rare adrenocortical carcinoma (ACC). Hormone production, size, spread and pathology are important context.", ["Abdominal or back pain, fullness, or a mass", "New features of excess cortisol or sex hormones, such as rapid weight change, weakness, or new hair/voice changes", "High blood pressure or low potassium from hormone excess"], "Hormone blood and urine tests, adrenal-protocol CT or MRI, and specialist endocrine assessment are central. Biopsy is not routine for every adrenal mass.", "Treatment is led by an adrenal cancer specialist and may involve surgery, mitotane, selected chemotherapy, radiation, and symptom control for hormone excess.", "Most ACC cannot be prevented. Genetic counselling may be relevant for rare inherited syndromes; a hormone-producing or enlarging adrenal mass needs specialist review."),
    "prostate_cancer": _profile("Most prostate cancers are adenocarcinomas. PSA, biopsy grade group, stage and clinical risk group guide decisions.", ["Early prostate cancer often has no symptoms", "Urinary change, blood in urine or semen, or bone pain need assessment but have many causes", "A new persistent urinary change should be discussed with a clinician"], "PSA is interpreted with history and examination; MRI and biopsy may be used. Imaging and pathology define grade and stage when cancer is diagnosed.", "Depending on risk and preferences, options can include active surveillance, surgery, radiation, hormone therapy, chemotherapy, targeted treatment, or radioligand treatment in selected advanced disease.", "Maintain general health, do not smoke, and discuss PSA testing based on age, ancestry, family history and local guidance."),
    "kidney_cancer": _profile("Kidney cancer often refers to renal-cell carcinoma. Imaging pattern, size, spread and kidney function help guide evaluation.", ["Blood in urine", "Persistent flank pain or a mass, although many tumors are found incidentally", "Unexplained weight loss, fatigue, or fever"], "Contrast CT or MRI of the kidneys, urine and blood tests, and staging imaging are used. Biopsy is selected when it will change management.", "Treatment may include active surveillance, partial or total nephrectomy, ablation, immunotherapy, targeted therapy, or radiation for symptom control in selected cases.", "Do not smoke, manage high blood pressure and healthy weight, avoid unnecessary toxin exposure, and discuss hereditary risk if multiple relatives are affected."),
    "ovarian_cancer": _profile("Ovarian cancer includes several epithelial and non-epithelial subtypes; symptoms may be vague and persistent.", ["Persistent bloating or increase in abdominal size", "Pelvic or abdominal pain, feeling full quickly, or appetite change", "New urinary urgency/frequency or unexplained weight change"], "Pelvic examination, ultrasound or CT, CA-125 in context, and surgery or biopsy for tissue diagnosis may be used.", "Treatment often combines surgery and chemotherapy; targeted therapy, maintenance treatment, immunotherapy trials, or genetic testing may be relevant by subtype.", "There is no population screening test for average-risk people. Discuss BRCA/Lynch-related family history, do not smoke, and seek review for persistent symptoms."),
    "cervical_cancer": _profile("Cervical cancer is often linked to persistent high-risk HPV infection and is largely preventable through vaccination and screening.", ["Abnormal vaginal bleeding, including after sex or after menopause", "Persistent watery or bloody discharge", "Pelvic pain or pain during sex"], "HPV testing, Pap/cervical screening, colposcopy and biopsy are used. Imaging helps stage a confirmed cancer.", "Treatment can include surgery for selected early disease, chemoradiation for many locally advanced cancers, and systemic treatment for recurrent or metastatic disease.", "HPV vaccination, recommended cervical screening, avoiding tobacco, and follow-up of abnormal screening results reduce risk."),
    "endometrial_cancer": _profile("Endometrial cancer begins in the uterine lining. Abnormal bleeding is an important symptom but can have many non-cancer causes.", ["Bleeding after menopause", "New heavy, irregular, or prolonged bleeding before menopause", "Pelvic pressure or pain in some cases"], "Pelvic examination, transvaginal ultrasound, endometrial biopsy or hysteroscopy, and pathology are used; imaging may help with staging.", "Treatment commonly includes surgery, with radiation, chemotherapy, hormone therapy, immunotherapy, or targeted treatment in selected situations.", "Maintain a healthy weight, manage diabetes or polycystic ovary syndrome with a clinician, and promptly assess abnormal uterine bleeding."),
    "stomach_cancer": _profile("Stomach cancer, also called gastric cancer, is evaluated with endoscopy and biopsy; location and biomarkers affect care.", ["Persistent indigestion, early fullness, or upper abdominal discomfort", "Unexplained weight loss, vomiting, or difficulty eating", "Black stool, anemia, or vomiting blood require urgent assessment"], "Upper endoscopy with biopsy is the core test. CT and selected biomarker tests, including HER2 or MSI testing, may guide staging and treatment.", "Treatment may combine surgery, chemotherapy, radiation in selected cases, HER2-targeted therapy, immunotherapy, nutrition support, and symptom management.", "Do not smoke, limit salt-preserved foods, treat H. pylori when diagnosed, and seek care for persistent or alarm digestive symptoms."),
    "esophageal_cancer": _profile("Esophageal cancer affects the swallowing tube; adenocarcinoma and squamous-cell carcinoma have different risk patterns.", ["Progressive difficulty or pain with swallowing", "Unintentional weight loss or food getting stuck", "Persistent reflux-like symptoms, chest discomfort, or hoarseness"], "Upper endoscopy with biopsy establishes diagnosis; CT, PET/CT, endoscopic ultrasound and nutrition assessment help stage disease.", "Treatment can include endoscopic therapy for selected early disease, surgery, chemotherapy, radiation, immunotherapy, targeted treatment, and swallowing/nutrition support.", "Avoid tobacco and heavy alcohol, manage chronic reflux with a clinician, maintain healthy weight, and discuss Barrett's esophagus surveillance if applicable."),
    "bladder_cancer": _profile("Bladder cancer is commonly urothelial carcinoma. Blood in urine needs evaluation even though there are many non-cancer causes.", ["Visible or microscopic blood in urine", "Burning, urgency, or frequent urination without infection", "Pelvic or back pain in more advanced situations"], "Urine testing, cystoscopy, imaging of the urinary tract and tissue sampling define the type and depth of a bladder tumor.", "Treatment may include endoscopic removal, medication placed in the bladder, surgery, chemotherapy, radiation, immunotherapy, or targeted therapy depending on stage.", "Do not smoke, reduce occupational chemical exposures, stay hydrated, and arrange prompt assessment of blood in urine."),
    "melanoma": _profile("Melanoma is a cancer of pigment-producing cells. A changing pigmented lesion needs skin examination and biopsy.", ["An asymmetric mole with irregular border, varied color, growth, bleeding, or itch", "A new dark or unusual spot", "A changing lesion under a nail or on palms, soles, or mucosal skin"], "Dermatology examination, dermoscopy and excisional biopsy are used. Sentinel node assessment and imaging depend on tumor depth and stage.", "Treatment can include wide local excision, node evaluation, immunotherapy, BRAF/MEK targeted treatment for eligible tumors, and monitoring.", "Avoid intense ultraviolet exposure and tanning beds, use sun protection, and check changing spots promptly."),
    "multiple_myeloma": _profile("Multiple myeloma is a plasma-cell cancer involving bone marrow; protein studies, kidney function and bone findings are important.", ["Persistent bone or back pain, fractures, or weakness", "Fatigue, recurrent infections, thirst, constipation, or confusion from high calcium", "Kidney problems or anemia"], "Blood and urine protein studies, CBC, kidney/calcium tests, bone imaging and bone-marrow biopsy establish diagnosis and risk features.", "Treatment may include combination medicines, steroids, targeted or immunotherapy, radiation for selected bone lesions, supportive bone/kidney care, and autologous transplant for eligible people.", "There is no proven general prevention. Promptly assess severe bone pain, neurological symptoms, infection, or kidney-related symptoms."),
    "hodgkin_lymphoma": _profile("Hodgkin lymphoma is a lymphatic cancer that requires tissue diagnosis and specialist staging.", ["A painless lymph-node swelling, often in the neck or chest", "Drenching night sweats, persistent fever, or unexplained weight loss", "Itching, fatigue, cough, or chest pressure in some people"], "Excisional lymph-node biopsy is preferred when possible; PET/CT, blood tests and pathology define subtype and stage.", "Treatment often includes chemotherapy with or without radiation; relapsed disease may use targeted or immunotherapy and transplant in selected cases.", "There is no reliable general prevention. Persistent enlarging lymph nodes or systemic symptoms need medical review."),
    "non_hodgkin_lymphoma": _profile("Non-Hodgkin lymphoma is a large family of lymphoid cancers, ranging from slow-growing to aggressive subtypes.", ["Persistent painless lymph-node swelling", "Fevers, drenching night sweats, unexplained weight loss, or fatigue", "Abdominal fullness, skin changes, or symptoms related to the affected organ"], "A good tissue biopsy with immunophenotyping and genetic studies is essential; PET/CT, blood tests and bone-marrow testing may be used for staging.", "Treatment varies by subtype and may include observation, chemotherapy, immunotherapy, targeted therapy, radiation, cellular therapy, or transplant.", "There is no routine prevention. Discuss immune suppression, infection risks, family history and persistent lymph-node swelling with a clinician."),
    "testicular_cancer": _profile("Testicular cancer often occurs in younger adults and has highly effective treatment pathways when promptly assessed.", ["A new painless testicular lump, enlargement, or firmness", "A feeling of heaviness or ache in the scrotum or lower abdomen", "Breast tenderness or back pain can occur in some situations"], "Clinical examination, scrotal ultrasound, blood tumor markers and CT staging are used. Surgery to remove the affected testicle provides diagnosis and treatment.", "Management may include surveillance, surgery, chemotherapy, radiation for selected seminomas, and specialized care for residual disease.", "There is no proven prevention. Testicular self-awareness and prompt review of a new lump or enlargement are important."),
    "gallbladder_cancer": _profile("Gallbladder cancer is uncommon and can be discovered during evaluation of gallstones or biliary symptoms.", ["Persistent upper-right abdominal pain or swelling", "Jaundice, itch, nausea, or appetite loss", "Unexplained weight loss or fever"], "Ultrasound, CT or MRI/MRCP, liver tests and tissue pathology when feasible are used to evaluate suspected disease.", "Treatment may include surgery when localized, systemic chemotherapy, immunotherapy or targeted treatment in selected cases, biliary drainage, and symptom support.", "There is no standard screening. Seek care for persistent jaundice or biliary symptoms and manage gallstone-related disease with a clinician."),
    "bile_duct_cancer": _profile("Bile duct cancer, also called cholangiocarcinoma, can obstruct bile flow. Type, location and molecular profile influence care.", ["Jaundice, dark urine, pale stool, or itch", "Upper abdominal discomfort, fever, or repeated biliary infections", "Unexplained weight loss or fatigue"], "Liver tests, ultrasound, CT or MRI/MRCP, endoscopic sampling and pathology are used; molecular testing may guide systemic treatment.", "Options include surgery for selected disease, biliary drainage, chemotherapy, immunotherapy or targeted treatment for eligible molecular changes, radiation in selected cases, and supportive care.", "There is no routine screening for average risk. Control chronic liver or bile duct disease with specialists and seek prompt care for jaundice or fever."),
    "sarcoma": _profile("Sarcoma is a diverse group of connective-tissue cancers. The exact tissue type must be confirmed by specialist pathology.", ["A growing lump, especially if deep, firm, or larger than about 5 cm", "Persistent unexplained bone or soft-tissue pain", "Reduced movement or nerve pressure symptoms depending on location"], "MRI or CT of the affected area and a planned core biopsy reviewed by a sarcoma specialist are important before definitive surgery.", "Care may combine specialist surgery, radiation, chemotherapy, targeted treatment, or clinical trials depending on subtype, grade and spread.", "Most sarcomas cannot be prevented. Arrange assessment of an enlarging or deep soft-tissue mass rather than attempting removal outside a specialist pathway."),
    "oral_cancer": _profile("Oral cancer can affect lips, tongue, gums, cheek, floor of mouth, or other mouth tissues.", ["A mouth sore that does not heal", "A persistent lump, pain, numbness, bleeding, or red/white patch", "Difficulty chewing, swallowing, or moving the jaw"], "Mouth and neck examination, dental or ENT review, imaging when needed, and biopsy of a concerning lesion.", "Treatment may use surgery, radiation, chemotherapy, immunotherapy or targeted therapy according to site and stage, with speech and nutrition support.", "Avoid tobacco and betel-nut exposure, limit alcohol, use sun protection for the lips, and attend review for persistent lesions."),
    "head_neck_cancer": _profile("Head and neck cancers arise in the throat, voice box, nose, sinuses, or nearby tissues; HPV and tobacco are relevant risks for some types.", ["Persistent hoarseness or sore throat", "A neck lump, swallowing difficulty, ear pain, or nasal blockage", "Unexplained weight loss or a mouth/throat lesion"], "ENT examination, endoscopy, imaging, HPV-related testing when relevant, and tissue biopsy are used.", "Treatment can combine surgery, radiation, chemotherapy, immunotherapy or targeted therapy, with dental, speech and nutrition support.", "Do not smoke or use smokeless tobacco, limit alcohol, consider HPV vaccination per local guidance, and seek review for persistent symptoms."),
    "anal_cancer": _profile("Anal cancer develops in the anal canal or nearby skin and has several possible cell types.", ["Anal bleeding, pain, itching, or a lump", "A change in bowel habits or narrowing of stool", "Swollen groin nodes or discharge"], "Examination, anoscopy, biopsy and CT/MRI/PET staging are used when cancer is diagnosed.", "Most anal squamous cancers are treated with combined chemoradiation; surgery may be used for persistent or recurrent disease.", "HPV vaccination, smoking cessation, safer sex, HIV care, and evaluation of persistent symptoms can reduce risk or improve early detection."),
    "small_intestine_cancer": _profile("Small-intestine cancers include adenocarcinoma, neuroendocrine tumors, lymphoma and sarcoma, so subtype matters.", ["Intermittent abdominal pain, nausea, or bowel obstruction symptoms", "Unexplained anemia, bleeding, or weight loss", "Diarrhea or flushing for some hormone-producing neuroendocrine tumors"], "CT/MR enterography, endoscopy or capsule studies, surgery/biopsy, and pathology define the tumor type.", "Treatment varies by subtype and can include surgery, chemotherapy, targeted treatment, somatostatin-related therapy, or radiation in selected cases.", "There is no standard population screening. Manage inherited syndromes or inflammatory bowel disease with specialists and assess persistent symptoms."),
    "appendix_cancer": _profile("Appendix tumors include neuroendocrine tumors and mucinous or other neoplasms; behavior varies widely.", ["Appendix tumors may be found during surgery for appendicitis", "Abdominal swelling, pain, change in bowel habit, or increased girth in some cases", "Many people have no specific symptoms"], "CT/MRI, surgical pathology and specialist review determine subtype and whether mucin has spread in the abdomen.", "Treatment may be observation, appendectomy, more extensive surgery, cytoreductive surgery with heated chemotherapy in selected situations, or systemic treatment depending on type.", "There is no routine prevention. A confirmed appendix tumor should be reviewed by a team experienced with its specific subtype."),
    "neuroendocrine_tumor": _profile("Neuroendocrine tumors can arise in many organs and may be slow-growing or aggressive; some release hormones.", ["Flushing, diarrhea, wheeze, or palpitations for some hormone-producing tumors", "Abdominal pain, weight change, or a mass effect", "Many neuroendocrine tumors are found incidentally"], "Specialized imaging, blood/urine hormone tests, pathology grade and receptor testing help define the tumor and extent.", "Treatment may include observation, surgery, somatostatin analogues, targeted medicines, peptide receptor radionuclide therapy, chemotherapy, or liver-directed treatment.", "Most cannot be prevented. A specialist should interpret hormone tests and determine the exact site and grade."),
    "bone_cancer": _profile("Primary bone cancers include osteosarcoma, Ewing sarcoma and other rare tumors; cancer that has spread to bone is a different situation.", ["Persistent bone pain that worsens or occurs at night", "A swelling or mass near a bone", "A fracture after minor injury or reduced movement"], "X-ray followed by MRI/CT and a carefully planned biopsy from a bone-tumor center are used before treatment.", "Treatment can include multi-agent chemotherapy, specialist limb-sparing surgery, radiation for selected subtypes, and rehabilitation.", "There is no general prevention. Persistent bone pain or an enlarging bony swelling needs prompt assessment."),
    "soft_tissue_sarcoma": _profile("Soft-tissue sarcoma arises in muscle, fat, blood vessels, nerves or other connective tissue.", ["An enlarging or deep soft-tissue lump", "Pain, pressure, or numbness if the mass presses on nearby structures", "Many early lumps are painless"], "MRI/CT and a planned core biopsy reviewed by a sarcoma specialist are important before surgery.", "Care may include wide surgical removal, radiation, chemotherapy or targeted treatment for selected subtypes, and specialist rehabilitation.", "No standard prevention exists. A lump that grows, is deep, painful, firm, or large should be assessed promptly."),
    "mesothelioma": _profile("Mesothelioma is a cancer of lining tissues, most often the pleura around the lungs; asbestos exposure is an important risk factor.", ["Progressive shortness of breath or chest pain", "Persistent cough or pleural fluid build-up", "Weight loss or fatigue"], "Chest imaging, fluid evaluation and specialist tissue biopsy are used; pathology markers distinguish mesothelioma from other cancers.", "Treatment can include surgery in selected cases, chemotherapy, immunotherapy, radiation for symptom control, fluid management and supportive care.", "Avoid asbestos exposure and follow workplace safety practices. Anyone with significant prior exposure and new respiratory symptoms should seek clinical review."),
    "thymic_cancer": _profile("Thymic tumors arise in the thymus in the chest and include thymoma and thymic carcinoma.", ["Chest discomfort, cough, shortness of breath, or swelling", "Symptoms from myasthenia gravis, such as drooping eyelids or muscle weakness, can occur with thymoma", "Some tumors are found incidentally on imaging"], "Chest CT/MRI, specialist pathology and assessment for associated autoimmune conditions are used.", "Treatment may include surgery, radiation, chemotherapy, immunotherapy in selected cases, and treatment of associated autoimmune disease.", "There is no known general prevention. New neuromuscular weakness or chest symptoms need medical review."),
    "salivary_gland_cancer": _profile("Salivary-gland cancers are uncommon and include multiple histologic subtypes with different behavior.", ["A persistent lump near the jaw, ear, or mouth", "Facial weakness, numbness, pain, or trouble opening the mouth", "A swelling that grows or becomes firm"], "Head and neck examination, ultrasound/CT/MRI and fine-needle or core biopsy reviewed by specialist pathology are used.", "Treatment usually involves surgery; radiation, targeted therapy, chemotherapy or other systemic treatment may be used for selected subtypes.", "There is no routine prevention. Persistent salivary-area swelling or facial nerve symptoms need prompt assessment."),
    "vulvar_cancer": _profile("Vulvar cancer affects external genital skin and has squamous and other subtypes.", ["Persistent itch, pain, burning, bleeding, or an open sore", "A new lump, thickening, color change, or wart-like lesion", "A lesion that does not improve with usual treatment"], "Pelvic examination and biopsy establish diagnosis; imaging is selected for staging or lymph-node assessment.", "Treatment can include surgery, radiation, chemotherapy or immunotherapy in selected cases, with symptom and sexual-health support.", "HPV vaccination, smoking cessation, treatment/follow-up of vulvar skin conditions, and prompt evaluation of persistent changes can reduce risk."),
    "vaginal_cancer": _profile("Vaginal cancer is uncommon and may be HPV-related or arise from other causes depending on subtype.", ["Abnormal vaginal bleeding or discharge", "Pelvic pain, pain with sex, or a vaginal mass", "Urinary or bowel symptoms in some situations"], "Pelvic examination, colposcopy, biopsy and imaging for confirmed disease are used.", "Treatment commonly uses radiation with or without chemotherapy; surgery may be selected for certain small or recurrent tumors.", "HPV vaccination, regular cervical screening, smoking cessation, and assessment of abnormal bleeding or discharge are important."),
    "uterine_sarcoma": _profile("Uterine sarcomas are uncommon connective-tissue cancers of the uterus and differ from endometrial carcinoma.", ["Abnormal uterine bleeding or bleeding after menopause", "A rapidly enlarging pelvic mass or pelvic pressure", "Pain, bloating, or urinary/bowel pressure symptoms"], "Pelvic imaging, surgery and specialist pathology are needed because imaging cannot reliably distinguish all uterine masses.", "Treatment often includes surgery; radiation, chemotherapy, hormone or targeted treatments depend on exact subtype and stage.", "There is no reliable prevention. New postmenopausal bleeding or rapidly enlarging uterine symptoms should be assessed promptly."),
    "urethral_cancer": _profile("Urethral cancer is rare and can arise from different cell types along the urethra.", ["Blood in urine or urethral bleeding", "A urethral lump, pain, or difficulty passing urine", "Repeated urinary infections or discharge without a clear cause"], "Urologic examination, cystoscopy/urethroscopy, biopsy and CT/MRI staging are used.", "Treatment may include surgery, radiation, chemotherapy or combined approaches depending on location and stage.", "There is no standard prevention. Prompt assessment of persistent urethral bleeding, obstruction, or a new mass is important."),
    "upper_urinary_tract_cancer": _profile("Upper urinary tract cancer includes urothelial tumors of the renal pelvis or ureter.", ["Blood in urine", "Flank pain or urinary obstruction symptoms", "Recurrent urinary infections or unexplained weight loss in some cases"], "CT urography, urine cytology, ureteroscopy with biopsy and kidney-function tests can be used.", "Treatment can include endoscopic approaches, kidney/ureter surgery, chemotherapy, immunotherapy or targeted treatment depending on risk and stage.", "Do not smoke, avoid certain occupational exposures, and arrange assessment of any visible blood in urine."),
    "spinal_cord_tumor": _profile("Spinal cord and spinal-column tumors may be primary or metastatic and can affect nerves through pressure.", ["Progressive back or neck pain, especially at night", "New weakness, numbness, walking difficulty, or bowel/bladder changes", "Pain radiating down an arm or leg"], "Neurological examination and contrast MRI are key. Biopsy or surgery may be necessary to establish the exact tumor type.", "Treatment can include surgery, radiation, systemic treatment according to tumor type, steroids only when prescribed, and rehabilitation.", "There is no general prevention. New weakness, numbness around the groin, or bowel/bladder loss is an emergency."),
    "peritoneal_cancer": _profile("Peritoneal cancer affects the lining of the abdomen and may be primary or related to another organ cancer.", ["Increasing abdominal size, bloating, or fluid build-up", "Abdominal pain, early fullness, nausea, or bowel changes", "Weight loss or fatigue"], "CT/MRI, fluid testing, tumor markers in context and tissue biopsy are used to identify the primary source and subtype.", "Treatment may include surgery, systemic chemotherapy, targeted treatment, or selected regional approaches depending on origin and spread.", "There is no routine prevention. Persistent abdominal enlargement, early fullness, or unexplained fluid build-up needs medical assessment."),
    "kaposi_sarcoma": _profile("Kaposi sarcoma is a vascular tumor associated with HHV-8 and immune-system context.", ["Purple, red-brown, or dark skin or mouth lesions", "Leg swelling or discomfort around lesions", "Breathlessness or digestive symptoms when internal organs are involved"], "Clinical examination, biopsy when needed, HIV testing/status review and assessment of immune function and organ involvement are used.", "Care may include effective HIV therapy when relevant, local treatment, chemotherapy, immunotherapy or radiation depending on extent and immune status.", "HIV prevention, early HIV testing and effective treatment, and specialist review of new vascular-looking lesions are important."),
    "burkitt_lymphoma": _profile("Burkitt lymphoma is a fast-growing B-cell non-Hodgkin lymphoma that needs urgent specialist evaluation.", ["Rapidly enlarging jaw, abdominal, or lymph-node mass", "Abdominal pain, bowel obstruction symptoms, or swelling", "Fever, night sweats, weight loss, or marked fatigue"], "Urgent tissue biopsy, flow cytometry, genetic testing, blood tests, PET/CT and bone-marrow/central-nervous-system assessment are used.", "Treatment requires urgent intensive hematology protocols with chemotherapy, targeted anti-CD20 therapy when appropriate, infection prevention and tumor-lysis monitoring.", "There is no general prevention. A rapidly enlarging mass or severe abdominal symptoms needs urgent medical evaluation."),
    "mantle_cell_lymphoma": _profile("Mantle-cell lymphoma is a B-cell non-Hodgkin lymphoma with variable behavior and specialized pathology markers.", ["Painless enlarged lymph nodes", "Fatigue, fevers, night sweats, or weight loss", "Abdominal fullness, bowel symptoms, or enlarged spleen"], "Tissue biopsy with immunophenotyping and genetic testing, PET/CT, blood tests and sometimes bone-marrow assessment define disease extent.", "Treatment may include observation for selected cases, chemoimmunotherapy, targeted BTK-inhibitor medicines, cellular therapy or transplant in selected relapsed disease.", "There is no standard prevention. Persistent or growing lymph-node swelling needs medical review."),
    "cutaneous_t_cell_lymphoma": _profile("Cutaneous T-cell lymphoma includes mycosis fungoides and Sézary syndrome and can resemble common inflammatory skin disease.", ["Persistent patches, plaques, or tumors on the skin", "Chronic itch, color change, scaling, or widespread redness", "Enlarged lymph nodes or systemic symptoms in more advanced disease"], "Repeated skin biopsies, dermatopathology review, blood tests/flow cytometry and staging imaging may be needed because early appearances can be nonspecific.", "Treatment may include skin-directed therapy, phototherapy, radiation, systemic immune/targeted medicines, or other specialist treatment based on stage.", "There is no proven prevention. A rash that persists or changes despite treatment should be reviewed by dermatology."),
    "chronic_lymphocytic_leukemia": _profile("Chronic lymphocytic leukemia (CLL) is a blood and bone-marrow cancer involving abnormal lymphocytes; some people are monitored before treatment is needed.", ["Often no symptoms at first", "Fatigue, repeated infections, swollen nodes, or abdominal fullness", "Fevers, night sweats, unintentional weight loss, or easy bruising"], "CBC, blood flow cytometry and genetic risk testing define CLL; imaging and bone-marrow tests are selected when needed.", "Many people start with active monitoring. When treatment is needed, targeted medicines, antibody treatment, chemotherapy in selected cases and supportive infection care may be used.", "There is no reliable prevention. Keep vaccinations and infection prevention current with the treating team and report persistent systemic symptoms."),
    "chronic_myeloid_leukemia": _profile("Chronic myeloid leukemia (CML) is a myeloid blood cancer commonly defined by the BCR-ABL1 genetic change.", ["Often no symptoms at first", "Fatigue, sweats, weight loss, or abdominal fullness from an enlarged spleen", "Easy bruising or infections in some cases"], "CBC, blood film, bone-marrow testing and BCR-ABL1 molecular testing establish diagnosis and guide monitoring.", "Targeted tyrosine-kinase inhibitor treatment is central for most CML; transplant is reserved for selected resistant or advanced cases.", "There is no proven general prevention. Regular molecular monitoring and medication review with hematology are important."),
}


_BANGALI_NAMES = {
    "breast_cancer": "স্তন ক্যান্সার", "lung_cancer": "ফুসফুসের ক্যান্সার", "brain_cancer": "মস্তিষ্কের টিউমার/ক্যান্সার", "liver_cancer": "লিভার ক্যান্সার", "pancreatic_cancer": "অগ্ন্যাশয়ের ক্যান্সার", "colon_cancer": "কোলন ক্যান্সার", "colorectal_cancer": "কোলোরেক্টাল ক্যান্সার", "skin_cancer": "ত্বকের ক্যান্সার", "thyroid_cancer": "থাইরয়েড ক্যান্সার", "blood_cancer": "রক্তের ক্যান্সার", "adrenal_cancer": "অ্যাড্রিনাল ক্যান্সার", "stomach_cancer": "পাকস্থলীর ক্যান্সার", "bile_duct_cancer": "পিত্তনালির ক্যান্সার", "melanoma": "মেলানোমা", "bone_cancer": "হাড়ের ক্যান্সার", "cutaneous_t_cell_lymphoma": "কিউটেনিয়াস টি-সেল লিম্ফোমা", "prostate_cancer": "প্রোস্টেট ক্যান্সার", "kidney_cancer": "কিডনি ক্যান্সার", "ovarian_cancer": "ডিম্বাশয়ের ক্যান্সার", "cervical_cancer": "জরায়ুমুখের ক্যান্সার", "endometrial_cancer": "এন্ডোমেট্রিয়াল ক্যান্সার",
}


def _section(title: str, content: str, bullets: list[str] | None = None) -> InformationSection:
    return InformationSection(title=title, content=content, bullets=bullets or [])


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
        return CancerTopic(key=key, label=label, registry_key=key if get_cancer_capability(key) else None)
    key = re.sub(r"[^a-z0-9]+", "_", lookup).strip("_") or "cancer"
    label = cleaned[0].upper() + cleaned[1:]
    return CancerTopic(key=key, label=label, registry_key=key if get_cancer_capability(key) else None)


def _sources(topic: CancerTopic) -> list[InformationSource]:
    query = quote_plus(topic.label)
    return [
        InformationSource(title="National Cancer Institute — Cancer Types", url="https://www.cancer.gov/types", source_type="government_reference"),
        InformationSource(title="National Cancer Institute — topic search", url=f"https://www.cancer.gov/search/results?swKeyword={query}", source_type="government_reference"),
        InformationSource(title="MedlinePlus — Cancer", url="https://medlineplus.gov/cancer.html", source_type="government_reference"),
        InformationSource(title="World Health Organization — Cancer", url="https://www.who.int/news-room/fact-sheets/detail/cancer", source_type="international_reference"),
    ]


def _question_intent(question: str) -> str:
    text = question.casefold()
    if any(term in text for term in ("treatment", "therapy", "medicine", "chemo", "চিকিৎসা", "ওষুধ")):
        return "treatment"
    if any(term in text for term in ("test", "diagnos", "biopsy", "scan", "পরীক্ষা", "নির্ণয়", "বায়োপসি")):
        return "tests"
    if any(term in text for term in ("prevent", "risk", "screen", "প্রতিরোধ", "ঝুঁকি", "স্ক্রিন")):
        return "prevention"
    if any(term in text for term in ("symptom", "sign", "লক্ষণ", "উপসর্গ")):
        return "symptoms"
    return "overview"


def _intent_sentence(profile: dict[str, Any], intent: str) -> str:
    if intent == "symptoms":
        return "Common warning symptoms are listed below, but they can also have non-cancer causes."
    if intent == "tests":
        return profile["tests"]
    if intent == "treatment":
        return profile["treatment"]
    if intent == "prevention":
        return profile["prevention"]
    return profile["overview"]


class CancerInformationService:
    """Return disease-specific education, with an optional provider enhancement."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _language(request: CancerInformationRequest) -> str:
        return request.language if request.language in {"bn", "en"} else ("bn" if _BENGALI.search(request.question) else "en")

    def answer(self, request: CancerInformationRequest) -> CancerInformationResponse:
        topic = normalize_topic(request.cancer)
        question = _clean_text(request.question, max_length=2400)
        language = self._language(request)
        try:
            dynamic = self._configured_model_answer(topic, question, language)
            if dynamic is not None:
                return dynamic
        except Exception:
            pass
        return self._offline_answer(topic, question, language)

    def _offline_answer(self, topic: CancerTopic, question: str, language: str) -> CancerInformationResponse:
        profile = _OFFLINE_PROFILES.get(topic.key)
        if profile is None:
            profile = _profile(
                f"{topic.label} is a specific cancer topic. Its exact subtype, location, stage, biomarkers and pathology determine how it is interpreted.",
                ["A persistent or changing symptom related to the affected body area", "Unexplained weight loss, fatigue, bleeding, or a growing lump", "A symptom that is severe, rapidly worsening, or affects function"],
                "A clinician chooses testing based on the affected area. Examination, imaging, blood tests, endoscopy and biopsy/pathology may be considered.",
                "Treatment is determined by the confirmed subtype, stage, biomarkers, overall health and care goals; it may include surgery, radiation or systemic treatment.",
                "Follow local screening advice, avoid tobacco, maintain general health, and discuss family history or persistent symptoms with a clinician.",
            )
        personal = bool(_PERSONAL_OR_DIAGNOSTIC.search(question))
        urgent = bool(_URGENT_TERMS.search(question))
        intent = _question_intent(question)
        if language == "bn":
            topic_bn = _BANGALI_NAMES.get(topic.key, topic.label)
            answer = (
                f"{topic_bn} সম্পর্কে এটি সাধারণ শিক্ষামূলক তথ্য। {profile['overview']} "
                "ব্যক্তিগত রোগ নির্ণয়, স্টেজ বা চিকিৎসার সিদ্ধান্তের জন্য মূল রিপোর্ট ও চিকিৎসকের মূল্যায়ন দরকার।"
            )
            sections = [
                _section("এটি কী", profile["overview"]),
                _section("সম্ভাব্য লক্ষণ", "এই লক্ষণগুলোর অনেক non-cancer কারণও থাকতে পারে; শুধু লক্ষণ দেখে ক্যান্সার নিশ্চিত করা যায় না।", profile["symptoms"]),
                _section("কীভাবে পরীক্ষা করা হয়", profile["tests"]),
                _section("চিকিৎসা সম্পর্কে সাধারণ ধারণা", profile["treatment"]),
                _section("ঝুঁকি কমানো ও প্রতিরোধ", profile["prevention"]),
            ]
            follow_up = ["লক্ষণ সম্পর্কে আরও জানতে চান?", "পরীক্ষা বা বায়োপসি কীভাবে হয় জানতে চান?", "চিকিৎসকের কাছে কোন প্রশ্ন করবেন জানতে চান?"]
            urgent_guidance = "শ্বাসকষ্ট, অজ্ঞান হওয়া, অতিরিক্ত রক্তপাত, নিয়ন্ত্রণহীন ব্যথা, বা ক্যান্সারের চিকিৎসার সময় উচ্চ জ্বর হলে দ্রুত স্থানীয় জরুরি সেবা নিন।"
        else:
            answer = f"{profile['overview']} {_intent_sentence(profile, intent)} This is educational information, not a diagnosis."
            sections = [
                _section("What it is", profile["overview"]),
                _section("Possible symptoms", "These symptoms can have many non-cancer causes; symptoms alone cannot confirm or exclude cancer.", profile["symptoms"]),
                _section("How doctors evaluate it", profile["tests"], ["Clinical history and examination", "Site-appropriate imaging or laboratory tests", "Biopsy and pathology when tissue confirmation is needed", "Stage and biomarker assessment when relevant"]),
                _section("Treatment overview", profile["treatment"]),
                _section("Prevention and risk reduction", profile["prevention"]),
            ]
            follow_up = ["Would you like symptoms, testing, treatment, or prevention information?", "Do you mean a specific subtype or stage?", "Would you like help preparing questions for a clinician?"]
            urgent_guidance = "Seek local emergency care for severe breathing difficulty, confusion or fainting, heavy bleeding, uncontrolled pain, or high fever during cancer treatment."
        if personal:
            answer += " A clinician must interpret personal symptoms, imaging and pathology in their full context."
        if urgent:
            answer += " Because the question includes a potentially urgent symptom, general information should not delay prompt medical assessment."
        return CancerInformationResponse(
            request_id=f"info-{uuid.uuid4().hex}", cancer=topic.label, answer=answer, sections=sections,
            follow_up_questions=follow_up, urgent_guidance=urgent_guidance, sources=_sources(topic),
            mode="safe_fallback", specialist_registry_match=bool(topic.registry_key),
            diagnostic_conclusion=False, expert_review_required=True,
        )

    def _configured_model_answer(self, topic: CancerTopic, question: str, language: str) -> CancerInformationResponse | None:
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        api_key = os.getenv("ONCOAEGIS_CANCER_AI_API_KEY") or gemini_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = (os.getenv("ONCOAEGIS_CANCER_AI_BASE_URL") or os.getenv("GEMINI_BASE_URL") or ("https://generativelanguage.googleapis.com/v1beta/openai" if gemini_key else "https://api.openai.com/v1")).rstrip("/")
        model = os.getenv("ONCOAEGIS_CANCER_AI_MODEL") or os.getenv("GEMINI_MODEL") or ("gemini-2.5-flash" if "generativelanguage.googleapis.com" in base_url else "gpt-4o-mini")
        language_name = "Bangla" if language == "bn" else "English"
        system = (
            "You are Onco Aegis AI, a careful cancer-information educator. Answer in " + language_name + ". "
            "Give general education only. Never diagnose, stage, rule out cancer, prescribe a drug/dose, or tell someone to start/stop treatment. "
            "Explain uncertainty and that pathology, history, examination and appropriate testing may be needed. "
            "Return strict JSON: answer, sections (objects with title, content, bullets), follow_up_questions, urgent_guidance."
        )
        user = json.dumps({"cancer_topic": topic.label, "question": question, "language": language_name, "reference_urls": [source.url for source in _sources(topic)]}, ensure_ascii=False)
        body = json.dumps({"model": model, "temperature": 0.15, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode("utf-8")
        request = Request(f"{base_url}/chat/completions", data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
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
                request_id=f"info-{uuid.uuid4().hex}", cancer=topic.label, answer=parsed["answer"], sections=sections,
                follow_up_questions=[str(item) for item in parsed.get("follow_up_questions", [])][:6],
                urgent_guidance=str(parsed.get("urgent_guidance", "Seek urgent local care for severe or rapidly worsening symptoms.")),
                sources=_sources(topic), mode="configured_ai", specialist_registry_match=bool(topic.registry_key),
                diagnostic_conclusion=False, expert_review_required=True,
            )
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, KeyError, IndexError, AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None


_service: CancerInformationService | None = None
_service_lock = threading.Lock()


def get_cancer_information_service() -> CancerInformationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = CancerInformationService()
    return _service
