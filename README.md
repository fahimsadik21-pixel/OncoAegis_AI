# Onco Aegis AI — LUNA16 + BUSI baselines

This repository now contains two reusable supervised imaging pipelines:

- **LUNA16:** case-safe train/validation split, on-demand 2D CT slices, binary
  lung segmentation, U-Net training and Dice/IoU evaluation.
- **BUSI:** class-stratified split, grayscale ultrasound loading, merged lesion
  masks, and a shared U-Net with segmentation plus normal/benign/malignant
  classification heads.

The loaders normalize masks to binary values. LUNA16 CT inputs are windowed to
`[-1000, 400] HU` and normalized to `[0, 1]`; the input is never multiplied by
the target mask.

LUNA16 also uses its local annotations.csv to create a case-separated
nodule-vs-no-annotated-nodule slice-classification baseline. These annotations
identify nodules; they do not provide benign/malignant pathology labels.

## Prepare LUNA16 arrays

From the project root:

```powershell
.venv\Scripts\python.exe -m src.preprocessing.ct_preprocess
```

## Verify data

```powershell
.venv\Scripts\python.exe -m src.data.test_dataset
```

## Train LUNA16

```powershell
.venv\Scripts\python.exe -m training.train_luna --epochs 5 --batch-size 4 --image-size 256
```

The command writes `luna16_unet_best.pt` and `luna16_unet_last.pt` under
`checkpoints/luna16/`. Add `--max-batches 2 --max-train-slices 32` for a quick
smoke run.

## Train LUNA16 nodule classifier

~~~powershell
.venv\Scripts\python.exe -m training.train_luna_nodule --epochs 10 --batch-size 8 --image-size 128 --patch-size 160
~~~

Evaluate and review the nodule classifier:

~~~powershell
.venv\Scripts\python.exe -m evaluation.evaluate_luna_nodule checkpoints/luna16_nodule/luna16_nodule_classifier_best.pt --image-size 128 --patch-size 160 --output-json outputs/luna16_nodule_report.json
.venv\Scripts\python.exe -m evaluation.visualize_luna checkpoints/luna16/luna16_unet_best.pt --image-size 64 --count 12
~~~

The nodule classifier is an annotation-detection baseline. LUNA16 does not
provide benign/malignant pathology labels, so its output must not be
interpreted as a cancer diagnosis.

## Train BUSI

```powershell
.venv\Scripts\python.exe -m training.train_busi --epochs 10 --batch-size 8 --image-size 256
```

The command writes `busi_multitask_unet_best.pt` and
`busi_multitask_unet_last.pt` under `checkpoints/busi/`. Add
`--max-batches 2 --max-train-samples 12` for a quick smoke run.

After training, create a detailed validation report and qualitative figures:

```powershell
.venv\Scripts\python.exe -m evaluation.evaluate_busi checkpoints/busi/busi_multitask_unet_best.pt --output-json outputs/busi_validation_report.json
.venv\Scripts\python.exe -m evaluation.visualize_busi checkpoints/busi/busi_multitask_unet_best.pt --count 12
```

The report includes a confusion matrix and normal/benign/malignant precision,
recall and F1. Each subsequent full training run also saves separate best
segmentation and best classification checkpoints.

## Use the BUSI model through the app

Start the API from the project root:

~~~powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
~~~

Upload one ultrasound image to POST /analyze/ultrasound. The response
contains the model class probabilities, predicted segmentation area and a
structured result marked for expert review. The endpoint never presents a
BUSI class as confirmed cancer and returns 503 if the checkpoint is absent.

## Use the CT specialist and multimodal routes

The DICOM engine can validate one CT series, build a 3D volume and run the
local LUNA16 lung-anatomy segmentation baseline:

```text
POST /analyze/ct-series
POST /analyze/ct-specialist
POST /analyze/multimodal/ct-series
```

The specialist route requires a CHEST series and reports segmented lung-mask
geometry and volume. It does not detect or confirm cancer. The multimodal
route can additionally accept one related PDF/TXT report and applies the same
pathology-versus-imaging conflict safeguards as the ultrasound route.

`GET /system/status` shows registered dataset state, checkpoint availability,
component readiness and platform safety gates.

## Cancer information assistant and web UI

The same FastAPI process serves a public, responsive educational assistant at
`/app`. It uses the sky-blue Onco Aegis AI interface and does not require an
account for a general question. The UI supports English/Bangla selection,
loading and error states, follow-up prompts, official reference links and an
optional local/OAuth account modal.

The information API is:

```text
POST /information/cancer
GET  /information/cancer/status
```

Example request:

```json
{
  "cancer": "lung cancer",
  "question": "What are common symptoms and how is it evaluated?",
  "language": "en"
}
```

Without a provider key, the API uses a bounded safe educational fallback. To
enable optional natural-language provider support, set
`ONCOAEGIS_CANCER_AI_API_KEY`, `ONCOAEGIS_CANCER_AI_BASE_URL` and
`ONCOAEGIS_CANCER_AI_MODEL` using the `.env.example` contract. Provider output
is still marked non-diagnostic and requires expert review; it must not replace
pathology, examination or a treating clinician.

## Authentication

The API now exposes a GitHub-style local account flow:

```text
POST /auth/register
POST /auth/login
POST /auth/refresh
POST /auth/logout
GET  /auth/me
GET  /auth/providers
```

Registration and login return a short-lived bearer access token plus a
rotating refresh token. Users and revocable sessions are stored in the separate
SQLite database `outputs/auth.sqlite3`; set `ONCOAEGIS_AUTH_DATABASE_PATH` to
change that location. Passwords are never stored in plaintext.

The Google and Apple “Continue with” flows are implemented at
`/auth/oauth/google/start` and `/auth/oauth/apple/start`. The callback exchanges
the one-time authorization code, verifies the provider-signed ID token, links
or creates the local user, and returns the same Onco Aegis AI session response as a
password login. Google uses its client ID/secret and redirect URI. Apple can
use `ONCOAEGIS_APPLE_CLIENT_SECRET`, or generate the client secret from
`ONCOAEGIS_APPLE_TEAM_ID`, `ONCOAEGIS_APPLE_KEY_ID` and
`ONCOAEGIS_APPLE_PRIVATE_KEY` (or `ONCOAEGIS_APPLE_PRIVATE_KEY_PATH`).

Set these values before enabling a provider:

```text
ONCOAEGIS_GOOGLE_CLIENT_ID
ONCOAEGIS_GOOGLE_CLIENT_SECRET
ONCOAEGIS_GOOGLE_REDIRECT_URI
ONCOAEGIS_APPLE_CLIENT_ID
ONCOAEGIS_APPLE_REDIRECT_URI
```

Apple's web callback uses `response_mode=form_post`; the API accepts that POST
callback as well as Google's query-parameter callback. Provider state is
single-use and verified before any account linking.

Set `ONCOAEGIS_JWT_SECRET_KEY` to a long, private deployment secret. The
development fallback is only intended for local testing. If the dependency is
not already installed in the environment, install it with:

```powershell
pip install "python-jose[cryptography]"
```

## Evaluate checkpoints

```powershell
.venv\Scripts\python.exe -m evaluation.evaluate_luna checkpoints/luna16/luna16_unet_best.pt --image-size 64 --output-json outputs/luna16_validation_metrics.json
.venv\Scripts\python.exe -m evaluation.evaluate_busi checkpoints/busi/busi_multitask_unet_best.pt
```

Metrics are validation metrics only. They are not clinical performance claims.

## Dataset registry and common sample format

The registry at `datasets/registry/datasets.json` tracks DeepLesion, FLARE,
BUSI and LUNA16 with modality, organ coverage, annotation type, access note,
local path, processing state and intended model. `DatasetRegistry` exposes
these entries to the backend, while `UniversalDataset` adapts supported
dataset outputs to the `StandardMedicalSample` contract. DeepLesion and FLARE
remain registered but are intentionally not downloaded yet.

## Specialist model registry and routing

The specialist registry is kept separate from the dataset registry. It records
the model ID, modality, organ, task, input contract, output contract,
checkpoint and research-safety status. The cancer registry maps a clinical
area to registered specialist model IDs; an empty list means the capability
is not implemented and must not be guessed.

The API exposes:

```text
GET  /models
GET  /cancer-capabilities
POST /route/model
```

A routing request can specify modality, organ, task and optional cancer type.
If several models match without a task, the router returns `task_required`.
If no specialist exists, it returns `specialist_model_unavailable`. Research
model routes always carry an expert-review safety status and never confirm
malignancy.

## Clinical document evidence and fusion

The document_intelligence module accepts PDF/TXT reports and returns limited
structured evidence such as pathology-confirmed terminology or
imaging-suspicious terminology. It does not return a cancer diagnosis;
scanned PDFs are marked as requiring OCR.

The fusion module combines imaging evidence with document evidence. Suspicious
imaging plus benign pathology produces an explicit conflict and withholds a
definitive conclusion. Imaging-only suspicion remains unconfirmed.

The API exposes POST /analyze/document and POST /analyze/fusion.

All imaging outputs are research outputs and require expert review. Imaging
alone never becomes a confirmed malignancy conclusion; explicit pathology
evidence is treated as evidence for review, and conflicting evidence causes
the fusion layer to withhold a definitive conclusion.
