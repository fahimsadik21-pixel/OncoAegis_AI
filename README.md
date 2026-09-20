# Onco Aegis AI

Onco Aegis AI is a research-oriented cancer information and medical-imaging
assistant. It combines a FastAPI backend, specialist inference pipelines,
document intelligence, evidence-aware reasoning, authenticated chat history,
and a responsive web interface.

The system is designed to explain findings and support discussion with a
qualified care team. It is not a clinical diagnostic device. A model output
never confirms malignancy by itself.

## What is implemented

- Cancer information assistant with English and Bangla responses.
- Local email/password authentication with JWT access and refresh sessions.
- Google and Apple OAuth-ready flows.
- User-scoped chat history with resume, private image thumbnails and deletion.
- PDF/TXT document intelligence for pathology, radiology and laboratory text.
- Multimodal evidence fusion with conflict detection and clinical abstention.
- One specialist API: `POST /analyze/specialist`.
- Model registry, dataset registry, routing and standard analysis contracts.
- Research-only safety fields, limitations, provenance and expert-review status.
- Responsive Onco Aegis AI web UI at `/app`.

## Specialist model catalogue

| Cancer area | Model | Type | Expected input | Main output |
|---|---|---|---|---|
| Lung | `luna16_lung_segmentation` | 2D U-Net | Preprocessed CT slice | Lung mask |
| Lung | `luna16_nodule_detector` | 2D patch classifier | Nodule-centred CT patch | Annotated-nodule likelihood |
| Breast | `busi_breast_segmentation` | 2D multi-task U-Net | Grayscale ultrasound image | Lesion mask and area |
| Breast | `busi_breast_classifier` | 2D classification head | Grayscale ultrasound image | Normal/benign/malignant research class |
| Brain | `msd_brain_tumor_segmentation` | 3D U-Net | Four-channel 3D MRI NIfTI | Tumor-region mask and voxel estimate |
| Pancreas | `msd_pancreas_segmentation` | 3D U-Net | 3D pancreas CT volume | Pancreas and region masks |
| Pancreas | `msd_pancreas_tumor_segmentation` | Two-stage 3D pipeline | 3D abdominal CT volume | Pancreas/tumor-region estimates |
| Colon | `msd_colon_tumor_segmentation` | 3D U-Net | 3D abdominal colon CT volume | Tumor-region estimate |
| Liver | `ircadb01_liver_tumor_segmentation` | Two-stage 3D pipeline | Liver CT DICOM series | Liver/tumor-region estimates |
| Skin | `isic2016_skin_lesion_segmentation` | 2D U-Net | RGB dermoscopy image | Lesion mask and pixel area |
| Thyroid | `tn3k_thyroid_nodule_segmentation` | 2D U-Net | Grayscale thyroid ultrasound | Nodule mask and bounding box |
| Blood | `cnmc2019_all_cell_classifier` | 2D CNN | RGB microscopy cell image | ALL-like/HEM-like research class |
| Blood/AML | `flowcap_aml_patient_classifier` | Structured pretrained model | Eight flow-cytometry CSV tubes | Patient/tube research scores |

The 3D models are intended for their original volume or DICOM series. The web
input layer may accept a raster scan export for transport compatibility, but
that path is explicitly a single-slice approximation and must not be treated
as reliable 3D detection or clinical interpretation.

## Datasets represented in the project

- **LUNA16:** lung CT and annotated-nodule research data. The nodule route
  identifies annotated nodule-like patches; LUNA16 does not provide pathology
  labels for benign/malignant diagnosis.
- **BUSI:** breast ultrasound images, masks and normal/benign/malignant
  research classes.
- **MSD Task01 BrainTumour:** four-channel brain MRI volumes and tumor masks.
- **MSD Task07 Pancreas:** pancreas CT volumes and tumor-region annotations.
- **MSD Task10 Colon:** abdominal colon CT volumes and tumor-region labels.
- **3D-IRCADb-01:** liver CT and liver/tumor segmentation resources.
- **ISIC 2016:** RGB dermoscopy images and skin-lesion masks.
- **TN3K:** thyroid ultrasound images and nodule masks.
- **C-NMC 2019:** microscopy images for ALL/HEM cell research classification.
- **DREAM6/FlowCAP-II:** eight-tube flow-cytometry AML research workflow.

Datasets, raw clinical files, trained checkpoints, generated outputs and local
databases are intentionally excluded from GitHub. They must be acquired and
validated locally according to their individual licenses and registry entries.

## Architecture

```text
Web UI (/app)
   -> FastAPI routes
   -> input validation and privacy filtering
   -> model router / orchestrator
   -> execution engine
   -> specialist service and adapter
   -> StandardAnalysisResult
   -> patient-facing explanation with safety boundaries
```

Important backend areas:

- `app/core/`: orchestration, routing, execution and input normalization.
- `app/imaging/`: modality-specific preprocessing and inference services.
- `app/models/`: model architectures.
- `app/adapters/`: conversion to the standard result contract.
- `app/clinical/`: document evidence, reasoning and multimodal fusion.
- `app/security/`: users, password hashing, JWT and OAuth support.
- `app/storage/`: privacy-aware case and chat storage.
- `app/registry/`: model, cancer capability and dataset registries.
- `web/`: frontend HTML, CSS, JavaScript and brand assets.

## Local setup

```powershell
cd U:\OncoAegis_AI
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app`.

The optional cancer-information provider uses:

```text
ONCOAEGIS_CANCER_AI_API_KEY=
ONCOAEGIS_CANCER_AI_BASE_URL=https://api.openai.com/v1
ONCOAEGIS_CANCER_AI_MODEL=gpt-4o-mini
```

Never commit `.env`, JWT secrets, OAuth secrets, patient data or API keys.

## Main API routes

```text
GET  /app
POST /information/cancer
GET  /information/cancer/status
POST /analyze/specialist
POST /analyze/document
POST /analyze/fusion
GET  /models
GET  /cancer-capabilities
POST /route/model
POST /auth/register
POST /auth/login
POST /auth/refresh
POST /auth/logout
GET  /auth/me
GET/POST/DELETE /chat/history
```

## Training and evaluation

Existing training and evaluation code is retained in `training/`,
`app/training/` and `evaluation/`. Example commands:

```powershell
\.venv\Scripts\python.exe -m training.train_luna --epochs 5 --batch-size 4 --image-size 256
\.venv\Scripts\python.exe -m training.train_busi --epochs 10 --batch-size 8 --image-size 256
\.venv\Scripts\python.exe -m evaluation.phase1_validation
```

Validation metrics are research-pipeline evidence only. They are not clinical
sensitivity, specificity, safety or prospective performance claims.

## Verification

```powershell
\.venv\Scripts\python.exe -m unittest discover -s tests -q
\.venv\Scripts\python.exe -m compileall -q app tests
node --check web\app.js
```

The test suite covers authentication, ownership boundaries, chat history,
document intelligence, DICOM validation, input contracts, model routing,
specialist integration and phase-one evaluation reporting.

## Deployment boundary

The frontend can be hosted on Vercel. The FastAPI backend and model inference
are better hosted on Render, Railway or a managed VM because PyTorch, medical
libraries and model checkpoints are not a good fit for Vercel serverless
limits. Configure the frontend API base URL to the deployed backend and set
production JWT, provider, CORS, retention and monitoring values before public
use.

## Safety and governance

This project is research software. Imaging outputs require qualified expert
review. Pathology, examination, clinical history and confirmatory testing are
outside the authority of an image model. Production use requires external
validation, privacy review, consent/retention policy, monitoring, incident
response and clinical governance approval.
