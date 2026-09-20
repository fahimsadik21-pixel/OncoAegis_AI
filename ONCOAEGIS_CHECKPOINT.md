# Onco Aegis Ai — Local Continuation Checkpoint

Updated: 2026-09-14

## Current state

Dataset acquisition is intentionally postponed. The dataset-independent CT
foundation has now been advanced through the integrated first working version
of Milestone 3B.

Previously completed:

- Python 3.12.10, `.venv`, RTX 3050 6GB, CUDA and PyTorch verification
- FastAPI core and capability registry
- Universal input inspector for DICOM, PDF, TXT, raster images and NIfTI
- PDF inspection and scanned-PDF OCR-needed detection
- Milestone 3A

## Completed in this work session

- DICOM series loader from a folder or uploaded `(filename, bytes)` entries
- Recursive file discovery with sidecar/non-DICOM skipping
- Same-modality, same-study, same-series validation
- Consistent dimensions, pixel spacing and orientation validation
- Slice sorting using ImagePositionPatient projection
- Conservative fallbacks to InstanceNumber, SliceLocation or filename
- 3D volume construction in `(slice, row, column)` order
- RescaleSlope/RescaleIntercept conversion to float32 CT values
- Safe technical metadata summary that excludes patient identifiers
- `POST /analyze/ct-series` endpoint for multiple DICOM uploads
- CT preprocessing primitives: optional resampling, HU windowing and float32
  normalization
- Conservative anatomy routing contract; body-region metadata does not select
  an organ or cancer model by itself
- Filename router fixed so `.dcm` and raster images are not misclassified

## Latest integration milestone

- DICOM loader → CT preprocessing → anatomy routing is now one pipeline
- `/analyze/ct-series` runs the full preparation path
- The endpoint reports original/preprocessed shape and spacing, HU window,
  clipped fraction, candidate organs and route status
- `CTPipelineResult.model_input` exposes the normalized volume and safe
  metadata while keeping raw DICOM datasets out of the model input object
- Current CT chest route remains conservative: candidate `lung`, selected
  organ `None`, route status `requires_anatomy_model`
- Existing LUNA16 lung-segmentation checkpoint connected to the CT pipeline
  through a lazy, batched specialist inference service
- Real local 171-slice LUNA16 volume completed the specialist smoke test on
  CUDA
- `POST /analyze/ct-specialist` and
  `POST /analyze/multimodal/ct-series` now expose safe CT specialist output
  and optional document fusion
- LUNA output reports lung-anatomy mask geometry and volume only; it does not
  claim nodule detection or cancer confirmation
- `/system/status` reports component readiness, checkpoint availability and
  safety gates
- Universal DICOM inspection no longer returns raw study, series or SOP UIDs
- Local CLI added: `python -m scripts.inspect_ct_series <dicom-folder>`
- The CLI emits a JSON-safe, non-diagnostic summary for real-folder checks

## Files added or changed

- `app/imaging/dicom/loader.py`
- `app/imaging/dicom/series.py`
- `app/imaging/dicom/metadata.py`
- `app/imaging/dicom/__init__.py`
- `app/imaging/ct/preprocessing.py`
- `app/imaging/ct/pipeline.py`
- `app/imaging/ct/luna_inference.py`
- `app/imaging/ct/__init__.py`
- `app/imaging/__init__.py`
- `app/core/anatomy_router.py`
- `app/core/router.py`
- `app/schemas/imaging.py`
- `app/main.py`
- `scripts/inspect_ct_series.py`
- `scripts/__init__.py`
- `tests/test_dicom_series.py`

## Verification

Run from `U:\OncoAegis_AI`:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m compileall -q app tests
```

Latest result: 37 tests passed, and compilation passed. The CLI help command
also runs successfully.

`pytest` is not installed in the current environment. The test suite uses
standard `unittest`, so no new package was required. Full HTTP client testing
was not used because this FastAPI environment asks for the optional `httpx2`
package; the async endpoint logic itself is covered directly.

## Dataset modeling milestone — LUNA16 and BUSI

Completed in the current work session:

- LUNA16 processed output regenerated for all 89 matched cases
- LUNA16 CT inputs remain intact normalized volumes; target masks are binary
  `0/1` and are not applied to the input
- Case-safe train/validation split and on-demand slice-level dataset
- LUNA16 DataLoader, compact 2D U-Net, mixed-precision training loop,
  checkpointing, Dice/IoU evaluation and evaluation CLI
- LUNA16 annotations.csv parser, nodule-centred patch dataset and balanced
  annotated-nodule vs no-annotated-nodule classifier
- LUNA16 nodule report with class-wise confusion matrix; corrected baseline
  reaches 75% annotated-nodule recall on the balanced validation split
- Full local LUNA16 segmentation baseline completed for one epoch at 64x64:
  validation Dice 0.8559 and IoU 0.8141; these are research pipeline metrics
  only, not clinical performance claims
- BUSI adapter for 780 images: 133 normal, 437 benign and 210 malignant
- BUSI multi-mask merging for images with more than one annotation
- BUSI class-stratified split, DataLoader and shared U-Net segmentation plus
  three-class classification model
- BUSI multi-task training loop, checkpointing and evaluation CLI
- BUSI detailed validation report with confusion matrix and per-class
  precision/recall/F1
- BUSI qualitative prediction visualization for ultrasound, ground truth,
  predicted mask and overlay
- Future BUSI runs save separate joint, segmentation-best and
  classification-best checkpoints
- BUSI checkpoint integrated into POST /analyze/ultrasound with lazy model
  loading, image preprocessing, segmentation/classification output and safe
  expert-review status
- Real BUSI image tested through the app endpoint successfully
- Dataset registry added for DeepLesion, FLARE, BUSI and LUNA16
- UniversalDataset and StandardMedicalSample contracts added
- GET /datasets exposes the non-sensitive registry
- Dataset/model/app tests added; all 37 unit tests pass and compilation passes
- DeepLesion and FLARE are registered as future datasets but remain
  intentionally not downloaded

## Specialist registry and routing milestone

Completed after the dataset milestone:

- Existing model registry hardened while preserving `get_all_models()` and
  `get_model()` compatibility
- Model metadata corrected to reflect actual 2D LUNA16 inputs and BUSI model
  heads
- Cancer registry now records model references, research status and safety
  notes
- Conservative specialist router added with modality/organ/task filtering,
  cancer-capability filtering, checkpoint availability checks and explicit
  unavailable/ambiguous states
- API endpoints added: `GET /models`, `GET /cancer-capabilities` and
  `POST /route/model`
- Router regression tests cover LUNA, BUSI, aliases, ambiguity and unsupported
  liver routing

The first registered specialist inference output is now connected to this
fusion contract through the LUNA16 CT path. No new cancer dataset is required
for the current registry, routing, DICOM, document-foundation or fusion work.

## Clinical intelligence milestone

Completed:

- Rule-based PDF/TXT document evidence extractor with source-type detection
  for pathology, radiology, laboratory and related reports
- OCR-needed propagation for low-text PDFs
- Explicit distinction between pathology-confirmed terminology and
  imaging-suspicious terminology
- Multimodal fusion contract for imaging plus document evidence
- Conflict detection that withholds a definitive conclusion when evidence
  disagrees
- Organ-aware evidence filtering to avoid cross-organ conflicts
- API endpoints added: POST /analyze/document, POST /analyze/fusion and
  POST /analyze/multimodal/ultrasound
- Regression coverage expanded to 37 passing tests

The document extractor is a deterministic foundation, not a substitute for
a validated medical NLP model. A validated NLP/OCR model, production UI and
new specialist datasets remain intentionally deferred until their capability
requirements are defined.

Useful commands from `U:\OncoAegis_AI`:

```powershell
.venv\Scripts\python.exe -m src.data.test_dataset
.venv\Scripts\python.exe -m training.train_luna --epochs 5 --batch-size 4 --image-size 256
.venv\Scripts\python.exe -m training.train_luna_nodule --epochs 10 --batch-size 8 --image-size 128 --patch-size 160
.venv\Scripts\python.exe -m training.train_busi --epochs 10 --batch-size 8 --image-size 256
.venv\Scripts\python.exe -m evaluation.evaluate_busi checkpoints\busi\busi_multitask_unet_best.pt --output-json outputs\busi_validation_report.json
.venv\Scripts\python.exe -m evaluation.visualize_busi checkpoints\busi\busi_multitask_unet_best.pt --count 12
```

The smoke checkpoints under `checkpoints\luna16` and `checkpoints\busi` only
prove that the end-to-end code path runs. They are not clinical or research
performance results. Full training and validation should be run before any
model-quality claim, and imaging output must remain conservative rather than
being presented as a cancer diagnosis.

## Specialist integration milestone — 2026-09-19

Completed in this work session:

- Registered liver, pancreas, colon, skin, blood, thyroid and brain cancer
  capabilities with explicit research-only safety notes.
- Completed the universal Orchestrator → ExecutionEngine → Specialist Service
  → Adapter → StandardAnalysisResult path for all 13 registered model IDs.
- Added the missing TN3K adapter and FlowCAP specialist wrapper.
- Added a dedicated LUNA16 annotated-nodule patch classifier service and
  adapter; it is not a malignancy classifier.
- Corrected output-field contracts for LUNA, Brain MRI, Skin, C-NMC and
  FlowCAP adapters.
- ExecutionEngine now forwards optional spacing metadata to `analyze` and
  `predict` services and records the selected registry model ID in result
  provenance.
- Repaired the dataset registry metadata contract and historical backup-file
  syntax so the complete local compile/test commands are clean.

Verification completed:

- 41 unittest tests passed.
- `.venv\Scripts\python.exe -m compileall -q app tests` passed.
- Local checkpoint smoke run passed for LUNA segmentation and nodule, BUSI,
  Brain MRI, both pancreas IDs, Colon, Liver, Skin, C-NMC and TN3K.
- Official DREAM6/FlowCAP Java-backed patient smoke run passed for the eight
  subject-180 tube CSVs.
- Every cancer capability route resolved to an available registered model.

Remaining boundaries:

- These are local research-pipeline and integration validations, not clinical
  validation, prospective evaluation or pathology-confirmed cancer evidence.
- DICOM/NIfTI/image/flow-cytometry upload endpoints are not yet unified behind
  one public specialist API; the current specialist core path is validated.
- Production authentication, audit/storage, UI and clinical governance remain
  next-phase work.

## Phase 2 — unified specialist API milestone — 2026-09-19

Completed:

- Added `POST /analyze/specialist` as one public entry point for registered
  specialists.
- Added safe upload normalization for CT DICOM series, NumPy/NIfTI volumes,
  raster images and eight-tube FlowCAP CSV input.
- Kept raw uploads in memory or a temporary workspace only; safe metadata
  excludes uploaded filenames and patient identifiers.
- Added model, modality, organ and optional cancer-route consistency checks.
- Added request-level API provenance to `StandardAnalysisResult` without
  claiming clinical diagnosis.
- Bumped the application version to `0.4.0`.

Verification:

- 45 unittest tests passed.
- `.venv\Scripts\python.exe -m compileall -q app tests` passed.
- Real BUSI checkpoint passed through the complete unified endpoint path.

Next Phase 2 milestone:

- Add durable, privacy-aware audit records and case storage with retention,
  access-control and deletion boundaries before building the UI.

## Phase 1 â€” model evaluation and validation milestone â€” 2026-09-19

Completed:

- Added `evaluation/phase1_validation.py`, a reproducible JSON-reporting
  evaluator covering every registered specialist model.
- Added explicit segmentation metrics (Dice, IoU, precision, recall,
  specificity, positive-case metrics), classification metrics (accuracy,
  macro-F1, balanced accuracy, AUROC, ECE, confusion matrix), checkpoint
  SHA-256 provenance and non-diagnostic safety fields.
- Added trainer-compatible deterministic splits and leakage audits. All
  completed evaluators report zero train/validation overlap.
- Added `tests/test_phase1_validation.py` and corrected the stale liver
  evaluation import.

Verification:

- Full held-out local evaluation: 12 models completed, 1 model marked
  `not_evaluable`, 0 evaluator failures.
- FlowCAP was correctly not scored because the local status CSV is not valid
  ground truth for the official model output.
- 50 unittest tests passed.
- `.venv\Scripts\python.exe -m compileall -q app src evaluation tests` passed.
- Report audit passed for all 13 registered model IDs with safe clinical
  fields and zero leakage flags for every completed evaluation.
- Reports are stored under `outputs\phase1_evaluation\` with one JSON report
  per model plus `phase1_summary.json`.

Important quality findings:

- Formal evaluation is now complete as an evidence pipeline, but this did not
  turn the models into clinically validated cancer detectors.
- Brain, pancreas, colon and liver tumor-region scores are currently weak and
  require model/data remediation before any production or clinical claim.
- External, prospective and pathology-linked validation remains required.

## Phase 2 follow-up - real medical input contract hardening - 2026-09-19

Completed:

- Added bounded upload handling to the unified specialist API: file-count,
  per-file, total-size and empty-upload limits are enforced before inference.
- Hardened the shared normalizer for CT DICOM, CT NumPy/NIfTI, MRI NumPy/NIfTI,
  raster imaging and FlowCAP CSV inputs with dimensional, numeric, finite-value
  and size validation.
- Added extensionless and `.ima`/`.dicom` CT-series detection, validated DICOM
  geometry and HU conversion metadata, and kept patient identifiers out of the
  normalized response metadata.
- Added NIfTI spacing provenance, explicit unknown-intensity marking for raw
  NumPy/NIfTI CT input, raster pixel limits and strict seven-channel FlowCAP
  schema/numeric validation.
- Preserved the existing Orchestrator -> ExecutionEngine -> Specialist Service
  -> Adapter -> StandardAnalysisResult architecture.

Verification:

- 54 unittest tests passed with the project `.venv` interpreter.
- Specialist-input and DICOM contract tests passed: 21 tests.
- `.venv\Scripts\python.exe -m compileall -q app src evaluation tests` passed.
- Bounded upload smoke check passed through the API upload reader.
- Unified `POST /analyze/specialist` smoke passed with the BUSI checkpoint,
  including `StandardAnalysisResult` and API provenance.

Boundaries carried forward:

- DICOM CT is converted to HU from its rescale metadata; NumPy/NIfTI CT input
  is intentionally marked as unknown intensity unless a trusted upstream
  conversion is supplied.
- This phase validates transport and input contracts, not diagnostic accuracy,
  pathology confirmation or clinical validation.
- Durable audit/case storage, authentication/authorization, retention and UI
  remain the next implementation phase.

## Authentication and authorization milestone - 2026-09-20

Completed:

- Added SQLite-backed user storage in `app/security/store.py`, separate from
  medical case storage, with local accounts, OAuth identity/state tables and
  revocable access/refresh sessions.
- Added salted PBKDF2-SHA256 password hashing with a stronger current format
  and verification compatibility for the earlier development hash format.
- Added signed JWT access and refresh tokens, refresh-token rotation, logout
  revocation, active-user lookup and bearer authentication dependencies.
- Unified role definitions and connected viewer, doctor, radiologist,
  researcher and administrator permissions to protected case/storage routes.
- Added `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`,
  `GET /auth/me` and `GET /auth/providers`.
- Added Google and Apple authorization-code exchange, signed ID-token
  verification against provider JWKS, verified-email account linking and
  OncoAegis session issuance. Google query callbacks and Apple's `form_post`
  callback are supported.
- Added focused authentication regression coverage in
  `tests/test_security_auth.py` and deployment notes in `README.md`.

Verification:

- 7 focused authentication tests passed.
- Full project suite: 68 unittest tests passed.
- `.venv\Scripts\python.exe -m compileall -q app tests` passed.
- FastAPI OpenAPI generation exposed the authentication routes successfully.

Security boundary:

- The local development JWT fallback must be replaced with
  `ONCOAEGIS_JWT_SECRET_KEY` before deployment.
- Google and Apple sign-in code is live in the API, but each provider remains
  unavailable until its deployment credentials and registered redirect URI are
  configured.

## Cancer information assistant and public product UI - 2026-09-20

Completed:

- Added a public `POST /information/cancer` endpoint for broad cancer-
  information questions in English or Bangla.
- Added a bounded, source-aware educational fallback that works without an
  external provider and accepts known as well as previously unregistered cancer
  topics.
- Added optional configured-model support through
  `ONCOAEGIS_CANCER_AI_API_KEY`, with strict JSON parsing and automatic safe
  fallback when the provider is unavailable.
- Added official reference links (NCI, MedlinePlus and WHO), urgent-care
  guidance, follow-up questions and explicit non-diagnostic/expert-review
  fields to every response.
- Added `web/index.html`, `web/styles.css` and `web/app.js` as a public-first,
  responsive Onco Aegis Ai assistant: sky-blue visual system with violet,
  coral and amber accents, animated CSS logo mark,
  subtle motion, English/Bangla control, loading/error states, references and
  optional local/OAuth account modal.
- Mounted the UI at `/app` and its assets at `/app/assets` in the existing
  FastAPI process; authentication remains available but is not required for a
  general educational question.
- Added the optional cancer-information environment contract to `.env.example`.

Verification:

- 6 focused cancer-information unittest tests passed.
- Full project suite: 74 unittest tests passed.
- `.venv\\Scripts\\python.exe -m compileall -q app tests` passed.
- Local Uvicorn smoke check passed for `/app`, `/app/assets/styles.css`,
  `/information/cancer/status` and `POST /information/cancer`.
- Smoke response confirmed `safe_fallback`, five sections and
  `diagnostic_conclusion: false`.

Boundaries:

- The built-in fallback is general education, not a diagnostic engine or a
  substitute for pathology, examination or clinician judgment.
- Natural-language coverage becomes broader when an approved provider key is
  configured; no provider credential is committed to the repository.
- Google and Apple sign-in remain structurally ready but unavailable until
  their real deployment credentials and redirect URIs are supplied.

## Next phase - product hardening and real-provider setup

- Run browser-based visual QA at desktop and mobile breakpoints and tune any
  spacing or accessibility findings.
- Configure and exercise the selected cancer-information provider in a
  controlled environment, including timeout, quota and content-review checks.
- Add authenticated conversation history only after retention, consent and
  deletion behavior are explicitly decided.
- Keep imaging/model outputs behind the existing expert-review and research-
  only boundaries; do not convert this educational assistant into a clinical
  diagnosis claim.

## UI refinement and language milestone - 2026-09-20

Completed:

- Replaced the broken logo asset reference with the generated Onco Aegis AI
  oncology mark and kept the animated shield/ribbon treatment.
- Unified text questions and specialist image uploads inside one assistant
  workspace; the old duplicate Step 1/Step 2 image panel is hidden.
- Added visible image routes for breast, brain, lung, liver, pancreas, colon,
  thyroid, blood and skin, mapped to the backend registry vocabulary.
- Added safe multipart submission to `POST /analyze/specialist` with sign-in
  gating, selected model metadata, bounded file selection and research-only
  result rendering in the main response card.
- Removed visible mojibake from the web UI and restored the Bangla language
  control with automatic Bangla detection when a user types Bangla.
- Added a clean Bangla fallback answer with sections, follow-up questions,
  urgent-care guidance and trusted source links.
- Completed desktop and 390px mobile browser QA for the public `/app` page.

Verification:

- 74 unittest tests passed with the project `.venv` interpreter.
- `.venv\\Scripts\\python.exe -m compileall -q app` passed.
- `node --check web\\app.js` passed.
- Live Bangla question returned Bangla answer sections and follow-ups.
- `/app` and `/app/assets/assets/onco-aegis-mark.png` returned HTTP 200.

Remaining product phases:

- Configure real Google/Apple and JWT deployment secrets, then exercise the
  provider callbacks in a controlled environment.
- Add authenticated conversation history only after retention, consent and
  deletion rules are explicitly chosen.
- Continue model-quality remediation for weak brain, pancreas, colon and
  liver evaluation results, followed by external/prospective validation.
- Perform production deployment hardening, monitoring, privacy review and
  clinical governance review. Research outputs must remain non-diagnostic.

## Authenticated chat history milestone - 2026-09-20

Completed:

- Added user-scoped SQLite conversation storage for text prompts and
  structured specialist-analysis summaries.
- Added authenticated `GET/POST/DELETE /chat/history` and
  `GET /chat/history/{conversation_id}` endpoints.
- Enforced ownership checks so one signed-in user cannot read another user's
  conversations; guest history requests return `401`.
- Kept uploaded medical files out of chat history; only bounded transcript and
  safe structured result metadata are persisted.
- Added a ChatGPT-style private history panel, New chat action, saved-chat
  list, conversation resume view and continuation on the same conversation.
- Connected both text answers and image-analysis results to the history save
  path, while keeping guest answers transient.

Verification:

- Full project suite: 77 unittest tests passed.
- `.venv\\Scripts\\python.exe -m compileall -q app tests` passed.
- `node --check web\\app.js` passed.
- Guest `/chat/history` request correctly returned HTTP 401.
- In-memory ownership, resume and deletion tests passed.

Current phase count:

- Completed: 10 major phases (CT foundation, datasets, specialist registry,
  clinical intelligence, unified specialist API, evaluation/input hardening,
  authentication, cancer assistant/UI, UI-language refinement, and
  authenticated chat history).
- Remaining: 4 product phases (OAuth/JWT production credentials, provider
  exercise, consent-based retention/deletion policy for longer-term history,
  and external model validation plus production/privacy/clinical governance).

## Final hardening pass - 2026-09-20

Completed without model retraining or UI colour changes:

- Fusion now returns evidence-first medical reasoning, explicit abstention and
  safe expert-review questions for imaging/document disagreement.
- PDF/TXT report analysis now exposes structured measurements, biomarkers,
  stage/TNM/date fields, extraction quality and OCR-required status.
- Signed-in specialist results and cases are owner-scoped; legacy storage is
  migrated safely with an `owner_user_id` column while raw uploads remain
  excluded from persistence.
- The main assistant now accepts one PDF/TXT report alongside image analysis,
  renders Bangla-safe structured feedback and saves the bounded result in the
  authenticated conversation history.
- Added visible report attachment support and removed newly introduced UI
  mojibake without changing the established colour system.

Verification:

- 77 unittest tests passed with `.venv\Scripts\python.exe`.
- `compileall` passed for the application.
- `node --check web\app.js` passed.

Still intentionally deferred: real Google/Apple deployment credentials,
provider quota/content review, model retraining/remediation, external clinical
validation, and production privacy/governance sign-off.
