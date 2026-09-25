const sessionKeys = {
  access: "oncoaegis.access_token",
  refresh: "oncoaegis.refresh_token",
};

const state = {
  accessToken: sessionStorage.getItem(sessionKeys.access),
  refreshToken: sessionStorage.getItem(sessionKeys.refresh),
  user: null,
  language: "en",
  providers: [],
  refreshing: false,
  analysisFiles: [],
  chatFiles: [],
  chatDocument: null,
  conversationId: null,
  conversationMessages: [],
  history: [],
  historyLoaded: false,
  loadingTimer: null,
  loadingStage: 0,
};

const analysisModels = {
  busi_breast_segmentation: { modality: "ultrasound", organ: "breast", task: "segmentation", label: "Ultrasound · Breast · Segmentation" },
  busi_breast_classifier: { modality: "ultrasound", organ: "breast", task: "classification", label: "Ultrasound · Breast · Classification" },
  isic2016_skin_lesion_segmentation: { modality: "dermatology", organ: "skin", task: "segmentation", label: "Dermatology · Skin · Segmentation" },
  msd_brain_tumor_segmentation: { modality: "mri", organ: "brain", task: "segmentation", label: "MRI · Brain · Segmentation" },
  luna16_lung_segmentation: { modality: "ct", organ: "lung", task: "segmentation", label: "CT · Lung · Segmentation" },
  luna16_nodule_detector: { modality: "ct", organ: "lung", task: "detection", label: "CT · Lung · Detection" },
};

Object.assign(analysisModels, {
  busi_breast_segmentation: { modality: "ULTRASOUND", organ: "breast", task: "segmentation", label: "Ultrasound / Breast / Segmentation" },
  busi_breast_classifier: { modality: "ULTRASOUND", organ: "breast", task: "classification", label: "Ultrasound / Breast / Classification" },
  tn3k_thyroid_nodule_segmentation: { modality: "ULTRASOUND", organ: "thyroid", task: "segmentation", label: "Ultrasound / Thyroid / Segmentation" },
  isic2016_skin_lesion_segmentation: { modality: "DERMOSCOPY", organ: "skin", task: "segmentation", label: "Dermoscopy / Skin / Segmentation" },
  msd_brain_tumor_segmentation: { modality: "MRI", organ: "brain", task: "segmentation", label: "MRI / Brain / Segmentation" },
  luna16_lung_segmentation: { modality: "CT", organ: "lung", task: "segmentation", label: "CT / Lung / Segmentation" },
  luna16_nodule_detector: { modality: "CT", organ: "lung", task: "lesion_detection", label: "CT / Lung / Lesion detection" },
  ircadb01_liver_tumor_segmentation: { modality: "CT", organ: "liver", task: "segmentation", label: "CT / Liver / Segmentation" },
  msd_pancreas_segmentation: { modality: "CT", organ: "pancreas", task: "segmentation", label: "CT / Pancreas / Segmentation" },
  msd_pancreas_tumor_segmentation: { modality: "CT", organ: "pancreas", task: "segmentation", label: "CT / Pancreas / Tumor-region analysis" },
  msd_colon_tumor_segmentation: { modality: "CT", organ: "colon", task: "segmentation", label: "CT / Colon / Tumor-region analysis" },
  cnmc2019_all_cell_classifier: { modality: "MICROSCOPY", organ: "blood", task: "classification", label: "Microscopy / Blood / Cell classification" },
  flowcap_aml_patient_classifier: { modality: "FLOW_CYTOMETRY", organ: "blood", task: "classification", label: "Flow cytometry / Blood / Classification" },
});

const imageInputContracts = {
  busi_breast_segmentation: {
    kind: "raster", accept: ".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp",
    extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    guidance: "Required: one grayscale breast ultrasound JPG/PNG/JPEG. Mammogram, CT, MRI, pathology slides, and phone photos are not valid BUSI inputs.",
  },
  busi_breast_classifier: {
    kind: "raster", accept: ".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp",
    extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    guidance: "Required: one grayscale breast ultrasound JPG/PNG/JPEG. Mammogram, CT, MRI, pathology slides, and phone photos are not valid BUSI inputs.",
  },
  tn3k_thyroid_nodule_segmentation: {
    kind: "raster", accept: ".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp",
    extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    guidance: "Required: one thyroid ultrasound JPG/PNG/JPEG. Upload an ultrasound image, not a CT, MRI, pathology slide, or general neck photo.",
  },
  isic2016_skin_lesion_segmentation: {
    kind: "raster", accept: ".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp",
    extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    guidance: "Required: one close, well-lit dermoscopy or skin-lesion JPG/PNG/JPEG. CT, MRI, ultrasound, and pathology images are not valid ISIC inputs.",
  },
  cnmc2019_all_cell_classifier: {
    kind: "raster", accept: ".png,.jpg,.jpeg,.webp,.bmp,image/png,image/jpeg,image/webp,image/bmp",
    extensions: [".png", ".jpg", ".jpeg", ".webp", ".bmp"],
    guidance: "Required: one RGB blood-cell microscopy JPG/PNG/JPEG. A blood report, CT, MRI, or ordinary photograph cannot be classified by this model.",
  },
  msd_brain_tumor_segmentation: {
    kind: "mri", accept: ".nii,.nii.gz,.npy,application/octet-stream",
    extensions: [".nii", ".nii.gz", ".npy"],
    guidance: "Required: one original 4-channel brain MRI .nii/.nii.gz NIfTI or 4D .npy volume. A JPG/PNG MRI screenshot is not a valid model input.",
  },
  luna16_lung_segmentation: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: full chest CT DICOM series, 3D CT .nii/.nii.gz NIfTI, or 3D .npy volume. A JPG/PNG CT screenshot is not a valid model input.",
  },
  luna16_nodule_detector: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: a nodule-centred chest CT source volume or DICOM series. A JPG/PNG screenshot is not a valid model input.",
  },
  ircadb01_liver_tumor_segmentation: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: full liver CT DICOM series, 3D CT .nii/.nii.gz NIfTI, or 3D .npy volume. JPG/PNG screenshots are not valid liver CT inputs.",
  },
  msd_pancreas_segmentation: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: full abdominal pancreas CT DICOM series, 3D CT .nii/.nii.gz NIfTI, or 3D .npy volume. JPG/PNG screenshots are not valid inputs.",
  },
  msd_pancreas_tumor_segmentation: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: full abdominal pancreas CT DICOM series, 3D CT .nii/.nii.gz NIfTI, or 3D .npy volume. JPG/PNG screenshots are not valid inputs.",
  },
  msd_colon_tumor_segmentation: {
    kind: "ct", accept: ".dcm,.dicom,.ima,.nii,.nii.gz,.npy,application/dicom,application/octet-stream",
    extensions: [".dcm", ".dicom", ".ima", ".nii", ".nii.gz", ".npy"],
    guidance: "Required: full abdominal colon CT DICOM series, 3D CT .nii/.nii.gz NIfTI, or 3D .npy volume. JPG/PNG screenshots are not valid inputs.",
  },
};

const DEFAULT_IMAGE_ACCEPT = "image/*,.dcm,.dicom,.ima,.nii,.nii.gz,.npy,.csv";
const MAX_CLIENT_IMAGE_FILES = 1024;

function fileSuffix(file) {
  const filename = String(file?.name || "").toLowerCase();
  if (filename.endsWith(".nii.gz")) return ".nii.gz";
  const index = filename.lastIndexOf(".");
  return index >= 0 ? filename.slice(index) : "";
}

function imageContract(modelId) {
  return imageInputContracts[modelId] || null;
}

function imageFilesError(modelId, files) {
  const contract = imageContract(modelId);
  const selected = [...files];
  if (!contract || !selected.length) return "";
  if (selected.length > MAX_CLIENT_IMAGE_FILES) {
    return `This route accepts up to ${MAX_CLIENT_IMAGE_FILES} image-series files at a time.`;
  }
  const unsupported = selected.find((file) => !contract.extensions.includes(fileSuffix(file)));
  if (unsupported) {
    return `“${unsupported.name}” is not a relevant input for this route. ${contract.guidance}`;
  }
  if (contract.kind !== "ct" && selected.length !== 1) {
    return `This route accepts exactly one file. ${contract.guidance}`;
  }
  if (contract.kind === "ct" && selected.length > 1) {
    const allDicom = selected.every((file) => [".dcm", ".dicom", ".ima", ""].includes(fileSuffix(file)));
    if (!allDicom) {
      return `Upload either one 3D NIfTI/NumPy volume or every slice from one CT DICOM series. ${contract.guidance}`;
    }
  }
  return "";
}

function applyImageContract(modelId, input, guidanceTarget = null) {
  const contract = imageContract(modelId);
  input.accept = contract?.accept || DEFAULT_IMAGE_ACCEPT;
  if (guidanceTarget) {
    guidanceTarget.textContent = contract?.guidance || "Choose an image route to see the exact trained input type.";
  }
}

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function setSession(payload) {
  if (!payload?.access_token || !payload?.refresh_token) return;
  state.accessToken = payload.access_token;
  state.refreshToken = payload.refresh_token;
  sessionStorage.setItem(sessionKeys.access, state.accessToken);
  sessionStorage.setItem(sessionKeys.refresh, state.refreshToken);
  state.user = payload.user || state.user;
  renderUser();
}

function clearSession() {
  state.accessToken = null;
  state.refreshToken = null;
  state.user = null;
  state.conversationId = null;
  state.conversationMessages = [];
  state.history = [];
  state.historyLoaded = false;
  sessionStorage.removeItem(sessionKeys.access);
  sessionStorage.removeItem(sessionKeys.refresh);
  $("#history-panel").hidden = true;
  renderUser();
}

function showToast(message) {
  const toast = $("#toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 3800);
}

function setServiceStatus(label, tone = "amber") {
  const target = $("#service-status");
  if (!target) return;
  target.innerHTML = `<span class="status-dot status-dot--${tone}" aria-hidden="true"></span>${escapeHtml(label)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeSourceUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" ? parsed.href : "#";
  } catch {
    return "#";
  }
}

function errorMessage(error) {
  if (error instanceof Error && error.message) {
    const message = error.message;
    if (message.includes("selected research model is unavailable")) {
      return "This model's trained checkpoint is not available on Railway yet. Your uploaded file may still be the correct type; use the route guidance above and try again after the checkpoint is published.";
    }
    if (message.toLowerCase().includes("memory")) {
      return "This analysis needs more hosting memory right now. Please try again shortly or choose a lighter image route.";
    }
    return message;
  }
  return "The request could not be completed. Please try again.";
}

async function api(path, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (state.accessToken && !headers.has("Authorization")) headers.set("Authorization", `Bearer ${state.accessToken}`);
  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch {
    throw new Error("Onco Aegis AI is not reachable right now. Check that the app is running and try again.");
  }
  if (response.status === 401 && retry && state.refreshToken && !path.includes("/auth/refresh")) {
    const refreshed = await refreshSession();
    if (refreshed) return api(path, options, false);
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg || "Invalid value").join(" ")
      : payload.detail;
    throw new Error(detail || "The request could not be completed.");
  }
  return payload;
}

function compactHistoryResponse(response, kind) {
  if (kind === "analysis") {
    return {
      analysis_id: response.analysis_id,
      specialist: response.specialist,
      findings: response.findings,
      measurements: response.measurements,
      limitations: response.limitations,
      conflicts: response.conflicts,
      safety: response.safety,
    };
  }
  return {
    request_id: response.request_id,
    cancer: response.cancer,
    answer: response.answer,
    sections: response.sections,
    follow_up_questions: response.follow_up_questions,
    urgent_guidance: response.urgent_guidance,
    sources: response.sources,
    mode: response.mode,
  };
}

function historyAssistantText(message) {
  try {
    const packet = JSON.parse(message.content || "{}");
    const response = packet.response || {};
    if (message.kind === "analysis") {
      return predictionSummary(response);
    }
    return response.answer || "Saved assistant response";
  } catch {
    return message.content || "Saved assistant response";
  }
}

function renderConversationThread() {
  const thread = $("#conversation-thread");
  if (!thread) return;
  thread.innerHTML = state.conversationMessages.map((message) => {
    const isUser = message.role === "user";
    const content = isUser ? message.content : historyAssistantText(message);
    const preview = !isUser && message.metadata?.preview_data_url
      ? `<img class="history-image-preview" src="${escapeHtml(message.metadata.preview_data_url)}" alt="Saved scan preview" />`
      : "";
    return `<div class="thread-message thread-message--${isUser ? "user" : "assistant"}"><small>${isUser ? "You" : "Onco Aegis AI"}</small>${preview}<p>${escapeHtml(content)}</p></div>`;
  }).join("");
  thread.hidden = !state.conversationMessages.length;
}

function renderHistoryList() {
  const list = $("#history-list");
  const empty = $("#history-empty");
  if (!list || !empty) return;
  empty.hidden = state.history.length > 0;
  list.innerHTML = state.history.map((item) => {
    const date = item.updated_at ? new Date(item.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "Saved chat";
    return `<div class="history-row"><button class="history-item${item.conversation_id === state.conversationId ? " is-selected" : ""}" type="button" data-history-id="${escapeHtml(item.conversation_id)}"><strong>${escapeHtml(item.title)}</strong><small>${date} · ${item.message_count} messages</small></button><label class="history-select"><input type="checkbox" data-history-select="${escapeHtml(item.conversation_id)}" aria-label="Select ${escapeHtml(item.title)}" /><span></span></label></div>`;
  }).join("");
  $$('[data-history-id]').forEach((button) => button.addEventListener("click", () => openConversation(button.dataset.historyId)));
  $$('[data-history-select]').forEach((box) => box.addEventListener("change", updateHistorySelection));
  updateHistorySelection();
  renderWorkspaceRail();
}

function renderWorkspaceRail() {
  const target = $("#rail-history");
  if (!target) return;
  if (!state.user) {
    target.innerHTML = "<p>Sign in to keep your conversations and image reviews together.</p>";
    return;
  }
  if (!state.history.length) {
    target.innerHTML = "<p>No saved conversations yet. Your next answer can stay here securely.</p>";
    return;
  }
  target.innerHTML = state.history.slice(0, 4).map((item) => {
    const date = item.updated_at
      ? new Date(item.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })
      : "Saved";
    return `<button type="button" data-rail-history-id="${escapeHtml(item.conversation_id)}"><span aria-hidden="true">◌</span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(date)}</small></button>`;
  }).join("");
  $$('[data-rail-history-id]').forEach((button) => {
    button.addEventListener("click", () => openConversation(button.dataset.railHistoryId));
  });
}

function updateHistorySelection() {
  const selected = $$('[data-history-select]:checked').length;
  const button = $("#delete-selected-history");
  if (button) { button.disabled = !selected; button.textContent = selected ? `Delete selected (${selected})` : "Delete selected"; }
}

async function deleteSelectedHistory() {
  const ids = $$('[data-history-select]:checked').map((box) => box.dataset.historySelect).filter(Boolean);
  if (!ids.length || !state.user) return;
  if (!window.confirm(`Delete ${ids.length} saved chat${ids.length > 1 ? "s" : ""}?`)) return;
  try {
    for (const id of ids) await api(`/chat/history/${encodeURIComponent(id)}`, { method: "DELETE" }, false);
    if (ids.includes(state.conversationId)) { state.conversationId = null; state.conversationMessages = []; $("#conversation-thread").hidden = true; }
    await loadHistory();
    showToast("Selected chats deleted.");
  } catch (error) { showToast(errorMessage(error)); }
}

function imagePreviewDataUrl(file) {
  if (!file || !file.type.startsWith("image/")) return Promise.resolve(null);
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const image = new Image();
      image.onload = () => {
        const scale = Math.min(1, 360 / Math.max(image.width, image.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(image.width * scale));
        canvas.height = Math.max(1, Math.round(image.height * scale));
        canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", .68));
      };
      image.onerror = () => resolve(null);
      image.src = reader.result;
    };
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

async function loadHistory() {
  if (!state.user) return;
  try {
    state.history = await api("/chat/history", {}, false);
    state.historyLoaded = true;
    renderHistoryList();
  } catch (error) {
    state.historyLoaded = false;
    showToast(errorMessage(error));
  }
}

async function openConversation(conversationId) {
  if (!conversationId || !state.user) return;
  try {
    const conversation = await api(`/chat/history/${encodeURIComponent(conversationId)}`, {}, false);
    state.conversationId = conversation.conversation_id;
    state.conversationMessages = Array.isArray(conversation.messages) ? conversation.messages : [];
    $("#response-section").hidden = false;
    $("#answer-card").hidden = true;
    $(".sources-card").hidden = true;
    $("#response-mode").textContent = "Saved conversation";
    renderConversationThread();
    renderHistoryList();
    $("#history-panel").hidden = true;
    $("#response-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showToast(errorMessage(error));
  }
}

async function persistTurn({ title, userMessage, response, kind, metadata = {} }) {
  if (!state.user) return;
  try {
    const conversation = await api("/chat/history", {
      method: "POST",
      body: JSON.stringify({
        conversation_id: state.conversationId,
        title,
        user_message: userMessage,
        assistant_message: JSON.stringify({ kind, response: compactHistoryResponse(response, kind) }),
        kind,
        metadata,
      }),
    }, false);
    state.conversationId = conversation.conversation_id;
    state.conversationMessages = Array.isArray(conversation.messages) ? conversation.messages : [];
    renderConversationThread();
    await loadHistory();
  } catch (error) {
    showToast(`Answer ready, but chat history could not be saved: ${errorMessage(error)}`);
  }
}

function startNewChat() {
  state.conversationId = null;
  state.conversationMessages = [];
  $("#cancer-topic").value = "";
  $("#question-text").value = "";
  $("#chat-files").value = "";
  $("#chat-document").value = "";
  $("#chat-model").value = "";
  state.chatFiles = [];
  state.chatDocument = null;
  renderChatFiles();
  renderChatDocument();
  updateCharacterCount();
  $("#response-section").hidden = true;
  $("#workspace").hidden = !state.user;
  $("#conversation-thread").hidden = true;
  $("#answer-card").hidden = false;
  $("#history-panel").hidden = true;
  renderHistoryList();
  $("#cancer-topic").focus();
}

function selectedAnalysisModel() {
  return analysisModels[$("#analysis-model").value] || analysisModels.busi_breast_segmentation;
}

function renderAnalysisAccess() {
  const signedIn = Boolean(state.user);
  const access = $("#analysis-access");
  const submit = $("#analysis-submit");
  if (!access || !submit) return;
  access.textContent = signedIn ? "Private workspace" : "Sign in required";
  access.classList.toggle("is-ready", signedIn);
  submit.disabled = !signedIn || !state.analysisFiles.length;
}

function humanizeKey(key) {
  return String(key || "").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function humanizeFinding(value) {
  if (value === null || value === undefined || value === "") return "No clear finding was returned.";
  if (Array.isArray(value)) return value.map((item) => humanizeFinding(item)).join(" ");
  if (typeof value !== "object") {
    return String(value)
      .replaceAll("breast_lesion_research_prediction", "breast image assessment")
      .replaceAll("research_prediction", "image assessment")
      .replaceAll("predicted_class", "primary image finding")
      .replaceAll("malignant", "higher-risk appearing")
      .replaceAll("benign", "lower-risk appearing")
      .replaceAll("normal", "no concerning pattern detected");
  }
  return Object.entries(value)
    .filter(([key]) => !["type", "dataset", "model_id", "task", "checkpoint", "provenance"].includes(key))
    .map(([key, item]) => {
      const friendlyKey = {
        predicted_class: "Overall pattern",
        class_probabilities: "Pattern confidence",
        lesion_area_fraction: "Highlighted area",
        tumor_volume_voxels: "Highlighted region size",
      }[key] || humanizeKey(key);
      if (typeof item === "number") {
        const number = item >= 0 && item <= 1 ? `${(item * 100).toFixed(1)}%` : Number(item).toLocaleString();
        return `${friendlyKey}: ${number}`;
      }
      return `${friendlyKey}: ${humanizeFinding(item)}`;
    })
    .join(". ");
}

function predictionSummary(response) {
  const first = Array.isArray(response.findings) ? response.findings[0] : null;
  const rawClass = first && typeof first === "object" ? first.predicted_class || first.class || first.label : null;
  if (!rawClass) return "The uploaded image was reviewed for patterns relevant to the selected cancer area. No single finding should be treated as a diagnosis.";
  const label = String(rawClass).toLowerCase();
  if (label === "benign") return "The image pattern appears lower-risk in this research assessment. This does not rule out disease and should be confirmed by a qualified clinician.";
  if (label === "malignant") return "The image pattern contains a higher-risk signal in this research assessment. This does not confirm cancer; prompt clinical review and confirmatory testing are important.";
  if (label === "normal") return "No concerning pattern was highlighted by this research assessment. A normal-looking result does not rule out disease or replace clinical review.";
  return `The research assessment highlighted: ${humanizeFinding(rawClass)}. A clinician must interpret this together with symptoms, history, and formal testing.`;
}

function friendlyDocumentFields(fields) {
  return Object.entries(fields || {}).map(([key, value]) => `${humanizeKey(key)}: ${humanizeFinding(value)}`);
}

function renderAnalysisFiles() {
  const list = $("#analysis-file-list");
  if (!list) return;
  const visibleFiles = state.analysisFiles.slice(0, 3);
  list.innerHTML = state.analysisFiles.length
    ? `${visibleFiles.map((file, index) => `<span class="file-chip"><span>${escapeHtml(file.name)}</span><small>${Math.ceil(file.size / 1024)} KB</small><button type="button" data-remove-analysis-file="${index}" aria-label="Remove ${escapeHtml(file.name)}">×</button></span>`).join("")}${state.analysisFiles.length > visibleFiles.length ? `<span class="file-chip"><span>+ ${state.analysisFiles.length - visibleFiles.length} more series files</span></span>` : ""}<button class="file-chip" type="button" data-clear-analysis-files>Clear all</button>`
    : "";
  $$('[data-remove-analysis-file]').forEach((button) => button.addEventListener("click", () => {
    state.analysisFiles.splice(Number(button.dataset.removeAnalysisFile), 1);
    $("#analysis-files").value = "";
    renderAnalysisFiles();
  }));
  $("[data-clear-analysis-files]")?.addEventListener("click", () => {
    state.analysisFiles = [];
    $("#analysis-files").value = "";
    $("#analysis-error").textContent = "";
    renderAnalysisFiles();
  });
  renderAnalysisAccess();
}

function setAnalysisFiles(files) {
  const selected = [...files];
  state.analysisFiles = selected.slice(0, MAX_CLIENT_IMAGE_FILES);
  renderAnalysisFiles();
  $("#analysis-error").textContent = selected.length > MAX_CLIENT_IMAGE_FILES
    ? `Only the first ${MAX_CLIENT_IMAGE_FILES} files were kept for this image series.`
    : imageFilesError($("#analysis-model").value, state.analysisFiles);
}

function renderChatFiles() {
  const list = $("#chat-file-list");
  if (!list) return;
  const visibleFiles = state.chatFiles.slice(0, 3);
  list.innerHTML = state.chatFiles.length
    ? `${visibleFiles.map((file, index) => `<span class="chat-file-chip"><span>${escapeHtml(file.name)}</span><small>${Math.ceil(file.size / 1024)} KB</small><button type="button" data-remove-chat-file="${index}">×</button></span>`).join("")}${state.chatFiles.length > visibleFiles.length ? `<span class="chat-file-chip"><span>+ ${state.chatFiles.length - visibleFiles.length} more series files</span></span>` : ""}<button class="chat-file-chip" type="button" data-clear-chat-files>Clear all</button>`
    : "";
  $$('[data-remove-chat-file]').forEach((button) => {
    button.textContent = "×";
    button.setAttribute("aria-label", "Remove attached image");
    button.addEventListener("click", () => {
      state.chatFiles.splice(Number(button.dataset.removeChatFile), 1);
      $("#chat-files").value = "";
      renderChatFiles();
    });
  });
  $("[data-clear-chat-files]")?.addEventListener("click", () => {
    state.chatFiles = [];
    $("#chat-files").value = "";
    $("#question-error").textContent = "";
    renderChatFiles();
  });
}

function setChatFiles(files) {
  const selected = [...files];
  state.chatFiles = selected.slice(0, MAX_CLIENT_IMAGE_FILES);
  renderChatFiles();
  $("#question-error").textContent = selected.length > MAX_CLIENT_IMAGE_FILES
    ? `Only the first ${MAX_CLIENT_IMAGE_FILES} files were kept for this image series.`
    : imageFilesError($("#chat-model").value, state.chatFiles);
}

function renderChatDocument() {
  const list = $("#chat-document-list");
  if (!list) return;
  list.innerHTML = state.chatDocument
    ? `<span class="chat-file-chip"><span>${escapeHtml(state.chatDocument.name)}</span><small>${Math.ceil(state.chatDocument.size / 1024)} KB</small><button type="button" data-remove-chat-document>×</button></span>`
    : "";
  const removeDocument = $("[data-remove-chat-document]");
  if (removeDocument) {
    removeDocument.textContent = "×";
    removeDocument.setAttribute("aria-label", "Remove attached report");
    removeDocument.addEventListener("click", () => {
      state.chatDocument = null;
      $("#chat-document").value = "";
      renderChatDocument();
    });
  }
}

function legacyRenderDocumentAnalysis(response, topic) {
  const fields = response.structured_fields || {};
  const fieldText = Object.entries(fields).map(([key, value]) => `${key}: ${analysisText(value)}`).join("\n");
  renderResponse({
    cancer: topic || response.document_type || "Clinical report",
    request_id: response.document_id,
    mode: "research_document_analysis",
    answer: "এই নথি থেকে কাঠামোবদ্ধ তথ্য বের করা হয়েছে। এটি শিক্ষামূলক সহায়তা; রোগ নির্ণয় বা চিকিৎসার সিদ্ধান্তের বিকল্প নয়।",
    urgent_guidance: "মূল রিপোর্টটি যোগ্য চিকিৎসক বা প্যাথলজিস্টের সঙ্গে পর্যালোচনা করুন।",
    sections: [
      { title: "Document", content: `${response.document_type || "Unknown"} - ${response.ocr_status || "text extracted"}`, bullets: [] },
      { title: "Extracted evidence", content: response.evidence?.length ? analysisText(response.evidence) : "No supported evidence phrase was found.", bullets: [] },
      { title: "Structured fields", content: fieldText || "No structured measurement, stage, or date was extracted.", bullets: [] },
      { title: "Quality and limits", content: JSON.stringify(response.extraction_quality || {}), bullets: [] },
    ],
    follow_up_questions: ["Which parts of this report need clinician confirmation?", "Is pathology or another confirmatory test available?"],
    sources: [],
  });
}

function analysisText(value) {
  if (value === null || value === undefined || value === "") return "Not provided";
  if (Array.isArray(value)) return value.map((item) => analysisText(item)).join("\n");
  if (typeof value === "object") {
    if (value.name) {
      const measured = value.value === null || value.value === undefined ? "not provided" : value.value;
      return `${value.name}: ${measured}${value.unit ? ` ${value.unit}` : ""}`;
    }
    return humanizeFinding(value);
  }
  return String(value);
}

function renderDocumentAnalysis(response, topic) {
  const reportType = humanizeKey(response.document_type || "clinical report");
  const evidence = Array.isArray(response.key_findings) && response.key_findings.length
    ? response.key_findings.join("\n")
    : (Array.isArray(response.evidence) ? response.evidence.map(humanizeFinding).join("\n") : "");
  const fields = friendlyDocumentFields(response.structured_fields || {}).join("\n");
  renderResponse({
    cancer: topic || reportType,
    request_id: response.document_id,
    mode: "report_review",
    answer: response.plain_language_summary || "Your report has been read and its available findings have been organized into a plain-language summary. This can help you prepare for a conversation with your care team; it cannot diagnose cancer or replace the original report.",
    urgent_guidance: "Please review the original report with the clinician who ordered the test. Ask them to explain any abnormal, suspicious, or unclear wording and whether follow-up testing is needed.",
    sections: [
      { title: "Report overview", content: `${reportType}. ${response.needs_ocr ? "This file needs OCR or a text-searchable copy before its wording can be safely explained." : "Text was available for structured review."}`, bullets: [] },
      { title: "Findings mentioned in the report", content: evidence || "No clearly supported finding phrase was extracted. The original report needs clinician review.", bullets: [] },
      { title: "Important report details", content: fields || "No stage, measurement, date, or other structured detail was extracted.", bullets: [] },
      { title: "What this means", content: "A report phrase can only be understood in context. Diagnosis and treatment decisions require the full report, examination, prior results, and the treating team.", bullets: [] },
    ],
    follow_up_questions: response.questions_for_care_team?.length ? response.questions_for_care_team : [
      "Which finding in this report is most important?", "Does this report suggest another test or biopsy?", "What should I ask my doctor about the risk or next step?",
    ],
    sources: [],
  });
}

function renderChatAnalysis(response, modelId, topic) {
  const specialist = response.specialist || {};
  const safety = response.safety || {};
  const model = analysisModels[modelId] || {};
  const findings = Array.isArray(response.findings) ? response.findings : [];
  const measurements = Array.isArray(response.measurements) ? response.measurements : [];
  const warnings = [
    ...(Array.isArray(response.limitations) ? response.limitations : []),
    ...(Array.isArray(response.conflicts) ? response.conflicts : []),
  ];
  renderResponse({
    cancer: topic || `${humanizeKey(model.organ || "medical")} image review`,
    request_id: response.analysis_id,
    mode: "research_analysis",
    answer: predictionSummary(response),
    urgent_guidance: safety.message || "Discuss this research output with the appropriate radiology, pathology, or hematology team before making any medical decision.",
    sections: [
      { title: "Detection summary", content: findings.length ? findings.map(humanizeFinding).join("\n") : "No clearly localised abnormal pattern was returned from this image.", bullets: [] },
      { title: "Highlighted region", content: measurements.length ? measurements.map(humanizeFinding).join("\n") : "This model did not return a reliable location or size from the uploaded image.", bullets: [] },
      { title: "What it may mean", content: "The result describes visual patterns in this image. It cannot by itself confirm cancer, identify stage, or determine treatment.", bullets: [] },
      { title: "Recommended next step", content: safety.message || "Show the original image and this summary to the appropriate specialist for formal interpretation.", bullets: [] },
      ...(warnings.length ? [{ title: "Important context", content: warnings.map(humanizeFinding).join("\n"), bullets: [] }] : []),
    ],
    follow_up_questions: [
      "What does this result measure, and what does it not measure?",
      "Which confirmatory tests or expert review should come next?",
      "What should I ask my radiologist, pathologist, or hematologist?",
    ],
    sources: [],
  });
  const previewFile = state.chatFiles.find((file) => file.type.startsWith("image/"));
  if (previewFile) renderScanPreview(previewFile, model.organ || "uploaded");
}

function renderScanPreview(file, organ) {
  const grid = $("#section-grid");
  if (!grid || !file) return;
  const url = URL.createObjectURL(file);
  const card = document.createElement("article");
  card.className = "scan-preview-card";
  card.innerHTML = `<div class="scan-preview-frame"><img src="${url}" alt="Uploaded ${escapeHtml(organ)} scan preview" /></div><div><span>IMAGE REVIEW</span><h3>Your uploaded image</h3><p>The model reviewed this image for ${escapeHtml(organ)}-related patterns. Exact boundaries are only reported when the selected model returns a valid localisation.</p></div>`;
  grid.prepend(card);
  card.querySelector("img").addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
}

function renderAnalysisResult(response) {
  const specialist = response.specialist || {};
  const safety = response.safety || {};
  const findings = Array.isArray(response.findings) ? response.findings : [];
  const measurements = Array.isArray(response.measurements) ? response.measurements : [];
  const limitations = Array.isArray(response.limitations) ? response.limitations : [];
  const cards = [
    ["Detection / prediction", findings.length ? findings.map((item) => escapeHtml(humanizeFinding(item))).join("<br />") : "No clear finding returned"],
    ["What was measured", measurements.length ? measurements.map((item) => escapeHtml(humanizeFinding(item))).join("<br />") : "No validated measurement returned"],
    ["Clinical meaning", "This is an AI-assisted research finding. It does not confirm cancer or replace a formal clinical review."],
    ["Next step", safety.message || "Discuss the result with the appropriate qualified care team."],
  ];
  const warnings = [...limitations, ...(Array.isArray(response.conflicts) ? response.conflicts : [])];
  $("#analysis-empty").hidden = true;
  $("#analysis-result").hidden = false;
  $("#analysis-status").textContent = "Image review completed";
  $("#analysis-result").innerHTML = `
    <div class="result-summary"><span class="result-check" aria-hidden="true">&#10003;</span><div><strong>Image review completed</strong><p>A patient-friendly summary is ready</p></div></div>
    <div class="result-grid">${cards.map(([label, value]) => `<div class="result-item"><span>${escapeHtml(label)}</span><div>${value}</div></div>`).join("")}</div>
    ${warnings.length ? `<div class="result-warning"><strong>Limitations and review notes</strong><ul>${warnings.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
  `;
}

async function submitAnalysis(event) {
  event.preventDefault();
  const error = $("#analysis-error");
  if (!state.user) {
    error.textContent = "Please sign in before uploading an image.";
    setModal(true);
    return;
  }
  if (!state.analysisFiles.length) {
    error.textContent = "Choose at least one supported file first.";
    return;
  }
  const model = selectedAnalysisModel();
  const inputError = imageFilesError($("#analysis-model").value, state.analysisFiles);
  if (inputError) {
    error.textContent = inputError;
    return;
  }
  const body = new FormData();
  body.append("model_id", $("#analysis-model").value);
  body.append("modality", model.modality);
  body.append("organ", model.organ);
  body.append("task", model.task);
  state.analysisFiles.forEach((file) => body.append("files", file));
  $("#analysis-submit").disabled = true;
  $("#analysis-submit span:first-child").textContent = "Analyzing...";
  $("#analysis-status").textContent = "Processing";
  error.textContent = "";
  try {
    const response = await api("/analyze/specialist", { method: "POST", body }, false);
    renderAnalysisResult(response);
    $("#analysis-result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (requestError) {
    $("#analysis-status").textContent = "Needs attention";
    error.textContent = errorMessage(requestError);
  } finally {
    $("#analysis-submit span:first-child").textContent = "Run image analysis";
    renderAnalysisAccess();
  }
}

function updateAnalysisModel() {
  const modelId = $("#analysis-model").value;
  const model = selectedAnalysisModel();
  const contract = imageContract(modelId);
  $("#analysis-meta").textContent = contract
    ? `${model.label} — ${contract.guidance}`
    : model.label;
  applyImageContract(modelId, $("#analysis-files"));
  $("#analysis-error").textContent = imageFilesError(modelId, state.analysisFiles);
}

async function refreshSession() {
  if (!state.refreshToken || state.refreshing) return false;
  state.refreshing = true;
  try {
    const payload = await api("/auth/refresh", {
      method: "POST",
      body: JSON.stringify({ refresh_token: state.refreshToken }),
    }, false);
    setSession(payload);
    return true;
  } catch {
    clearSession();
    return false;
  } finally {
    state.refreshing = false;
  }
}

function initials(user) {
  const value = user?.username || user?.email || "OA";
  return value.split(/[\s._@-]+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "OA";
}

function renderUser() {
  const signedIn = Boolean(state.user);
  $("#access-gateway").hidden = signedIn;
  $("#workspace").hidden = !signedIn;
  $("#open-auth").hidden = signedIn;
  $("#open-history").hidden = !signedIn;
  $("#refresh-workspace").hidden = !signedIn;
  $("#user-button").hidden = !signedIn;
  if (signedIn) {
    const name = state.user.username || state.user.email || "Account";
    $("#user-avatar").textContent = initials(state.user);
    $("#user-label").textContent = "Sign out";
  }
  renderAnalysisAccess();
  renderWorkspaceRail();
  if (signedIn && !state.historyLoaded) loadHistory();
}

function setModal(open) {
  const modal = $("#auth-modal");
  modal.hidden = !open;
  document.body.classList.toggle("is-modal-open", open);
  if (open) {
    window.setTimeout(() => $("#email").focus(), 60);
  }
}

function setAuthMode(mode) {
  const register = mode === "register";
  $$(`[data-auth-mode]`).forEach((button) => button.classList.toggle("is-active", button.dataset.authMode === mode));
  $("#username-field").hidden = !register;
  $("#auth-title").textContent = register ? "Create your account" : "Welcome back";
  $("#auth-subtitle").textContent = register ? "Keep your workspace identity ready when you need it." : "Save your session for a more personal workspace.";
  $("#auth-submit span:first-child").textContent = register ? "Create account" : "Sign in";
  $("#password").autocomplete = register ? "new-password" : "current-password";
  $("#auth-form").dataset.mode = mode;
  $("#auth-message").textContent = "";
}

function updateCharacterCount() {
  const value = $("#question-text").value;
  $("#character-count").textContent = `${value.length} / 2400`;
}

const loadingStages = [
  "Understanding clinical context",
  "Reviewing uploaded information",
  "Checking relevant evidence",
  "Preparing a clear explanation",
];

function renderLoadingStages() {
  const steps = $$("#loading-steps li");
  steps.forEach((step, index) => {
    step.classList.toggle("is-active", index === state.loadingStage);
    step.classList.toggle("is-complete", index < state.loadingStage);
  });
  const active = loadingStages[state.loadingStage];
  if (active) $("#loading-copy").textContent = `${active}…`;
}

function setLoading(loading, title = "Reading your question", copy = "Building a careful, general explanation...") {
  $("#ask-button").disabled = loading;
  $("#loading-state").hidden = !loading;
  $("#question-form").classList.toggle("is-loading", loading);
  document.body.classList.toggle("is-processing", loading);
  window.clearInterval(state.loadingTimer);
  if (loading) {
    $("#loading-title").textContent = title;
    $("#loading-copy").textContent = copy;
    $("#question-error").textContent = "";
    state.loadingStage = 0;
    window.setTimeout(renderLoadingStages, 0);
    state.loadingTimer = window.setInterval(() => {
      state.loadingStage = (state.loadingStage + 1) % loadingStages.length;
      renderLoadingStages();
    }, 2400);
  } else {
    state.loadingTimer = null;
  }
}

function renderResponse(response) {
  document.body.classList.add("viewing-result");
  $("#workspace").hidden = true;
  $("#response-section").hidden = false;
  $("#answer-card").hidden = false;
  $("#answer-topic").textContent = response.cancer || "Cancer information";
  $("#answer-id").textContent = response.request_id ? `Reference ${response.request_id.slice(-8)}` : "Reference ready";
  $("#answer-lead").textContent = response.answer || "No explanation was returned.";
  $("#response-mode").textContent = response.mode === "configured_ai"
    ? "Source-aware AI guide"
    : response.mode === "research_analysis"
      ? "Detection / prediction review"
      : response.mode === "report_review"
        ? "Clinical report review"
      : "Safe educational guide";
  $("#urgent-copy").textContent = response.urgent_guidance || "Seek local care for severe or rapidly worsening symptoms.";

  const sections = Array.isArray(response.sections) ? response.sections : [];
  $("#section-grid").innerHTML = sections.map((section) => {
    const bullets = Array.isArray(section.bullets) && section.bullets.length
      ? `<ul>${section.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
      : "";
    return `<article class="info-section"><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.content)}</p>${bullets}</article>`;
  }).join("") || `<article class="info-section"><h3>General context</h3><p>Bring this topic to a qualified care team for information specific to your situation.</p></article>`;

  const followUps = Array.isArray(response.follow_up_questions) ? response.follow_up_questions : [];
  $("#follow-up-list").innerHTML = followUps.slice(0, 6).map((question) => `<button class="follow-up-button" type="button" data-follow-up="${escapeHtml(question)}">${escapeHtml(question)}</button>`).join("");
  $$("[data-follow-up]").forEach((button) => button.addEventListener("click", () => {
    $("#question-text").value = button.dataset.followUp || "";
    updateCharacterCount();
    $("#question-text").focus();
    window.scrollTo({ top: $("#assistant-title").getBoundingClientRect().top + window.scrollY - 110, behavior: "smooth" });
  }));

  const sources = Array.isArray(response.sources) ? response.sources : [];
  $("#source-list").innerHTML = sources.map((source) => `<a class="source-link" href="${safeSourceUrl(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title || "Reference")}</a>`).join("");
  $(".sources-card").hidden = !sources.length;
  window.setTimeout(() => { $("#response-section").scrollIntoView({ behavior: "smooth", block: "start" }); }, 80);
}

function showWorkspace() {
  document.body.classList.remove("viewing-result");
  $("#response-section").hidden = true;
  $("#workspace").hidden = !state.user;
  $("#analysis-section").hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function submitQuestion(event) {
  event.preventDefault();
  const topic = $("#cancer-topic").value.trim();
  const question = $("#question-text").value.trim();
  const modelId = $("#chat-model").value;
  const hasFiles = state.chatFiles.length > 0;
  const hasDocument = Boolean(state.chatDocument);
  const answerLanguage = state.language === "bn" || /[\u0980-\u09ff]/.test(`${topic} ${question}`) ? "bn" : "en";
  const error = $("#question-error");
  if (topic.length > 120) { error.textContent = "Please keep the cancer type or topic under 120 characters."; $("#cancer-topic").focus(); return; }
  if (question.length > 2400) { error.textContent = "Please keep the question under 2400 characters."; return; }

  if (hasDocument && hasFiles) {
    error.textContent = "Please attach an image or a report at a time.";
    return;
  }

  if (hasDocument) {
    if (!state.user) {
      error.textContent = "Please sign in before uploading a report to your private workspace.";
      setModal(true);
      return;
    }
    const body = new FormData();
    body.append("file", state.chatDocument);
    setLoading(true, "Reading your report", "Extracting safe structured evidence...");
    try {
      const response = await api("/analyze/document", { method: "POST", body }, false);
      renderDocumentAnalysis(response, topic);
      await persistTurn({ title: topic || "Clinical report", userMessage: question || "Report analysis", response, kind: "analysis", metadata: { document_type: response.document_type } });
    } catch (requestError) {
      error.textContent = errorMessage(requestError);
    } finally {
      setLoading(false);
    }
    return;
  }

  if (hasFiles) {
    if (!state.user) {
      error.textContent = "Please sign in before uploading an image to your private workspace.";
      setModal(true);
      return;
    }
    if (!modelId || !analysisModels[modelId]) {
      error.textContent = "Choose the image route that matches your file before continuing.";
      $("#chat-model").focus();
      return;
    }
    const model = analysisModels[modelId];
    const inputError = imageFilesError(modelId, state.chatFiles);
    if (inputError) {
      error.textContent = inputError;
      return;
    }
    const body = new FormData();
    body.append("model_id", modelId);
    body.append("modality", model.modality);
    body.append("organ", model.organ);
    body.append("task", model.task);
    if (topic) body.append("cancer_type", topic);
    state.chatFiles.forEach((file) => body.append("files", file));
    setLoading(true, "Reviewing your uploaded file", "Preparing a structured research summary...");
    try {
      const response = await api("/analyze/specialist", { method: "POST", body }, false);
      renderChatAnalysis(response, modelId, topic);
      await persistTurn({
        title: topic || model.label,
        userMessage: question || `Image analysis request for ${model.label}`,
        response,
        kind: "analysis",
        metadata: { model_id: modelId, modality: model.modality, organ: model.organ, task: model.task, preview_data_url: await imagePreviewDataUrl(state.chatFiles[0]) },
      });
    } catch (requestError) {
      error.textContent = errorMessage(requestError);
    } finally {
      setLoading(false);
    }
    return;
  }

  if (!topic) { error.textContent = "Please enter a cancer type or topic."; $("#cancer-topic").focus(); return; }
  if (question.length < 3) { error.textContent = "Please write a little more about what you would like to know."; $("#question-text").focus(); return; }
  setLoading(true);
  try {
    const response = await api("/information/cancer", { method: "POST", body: JSON.stringify({ cancer: topic, question, language: answerLanguage }) }, false);
    renderResponse(response);
    await persistTurn({ title: topic, userMessage: question, response, kind: "text", metadata: { language: answerLanguage, topic } });
  } catch (requestError) {
    error.textContent = errorMessage(requestError);
  } finally {
    setLoading(false);
  }
}

function selectPrompt(button) {
  $$(".prompt-chip").forEach((item) => item.classList.remove("is-selected"));
  button.classList.add("is-selected");
  $("#question-text").value = button.dataset.prompt || "";
  updateCharacterCount();
  $("#question-text").focus();
}

function selectLanguage(language) {
  state.language = language === "bn" ? "bn" : "en";
  $$("[data-language]").forEach((button) => button.classList.toggle("is-active", button.dataset.language === state.language));
}

function selectCoverage(modelId) {
  const model = analysisModels[modelId];
  if (!model) {
    applyImageContract("", $("#chat-files"), $("#upload-guidance"));
    return;
  }
  $("#chat-model").value = modelId;
  applyImageContract(modelId, $("#chat-files"), $("#upload-guidance"));
  $$('[data-coverage-model]').forEach((button) => button.classList.toggle("is-selected", button.dataset.coverageModel === modelId));
  $("#question-error").textContent = imageFilesError(modelId, state.chatFiles);
  $("#chat-model").focus();
}

function renderProviders() {
  const providerMap = Object.fromEntries((state.providers || []).map((item) => [item.provider, item]));
  $$('[data-provider]').forEach((button) => {
    const provider = button.dataset.provider;
    const item = providerMap[provider];
    const enabled = Boolean(item?.enabled);
    button.disabled = !enabled;
    button.querySelector("small").textContent = enabled ? "Continue" : "Configure later";
    button.onclick = enabled ? () => beginProviderLogin(provider) : () => showToast(`${provider[0].toUpperCase() + provider.slice(1)} sign-in is waiting for server credentials.`);
  });
}

async function loadProviders() {
  try {
    state.providers = await api("/auth/providers", {}, false);
    renderProviders();
  } catch {
    $$('[data-provider] small').forEach((item) => { item.textContent = "Unavailable"; });
  }
}

async function beginProviderLogin(provider) {
  try {
    const payload = await api(`/auth/oauth/${encodeURIComponent(provider)}/start`, {}, false);
    if (payload.authorization_url) window.location.assign(payload.authorization_url);
  } catch (error) { showToast(errorMessage(error)); }
}

async function submitAuth(event) {
  event.preventDefault();
  const mode = $("#auth-form").dataset.mode || "login";
  const message = $("#auth-message");
  const button = $("#auth-submit");
  const body = { email: $("#email").value.trim(), password: $("#password").value };
  if (mode === "register") body.username = $("#username").value.trim() || undefined;
  message.textContent = "";
  button.disabled = true;
  button.querySelector("span:first-child").textContent = mode === "register" ? "Creating account…" : "Signing in…";
  try {
    const payload = await api(`/auth/${mode}`, { method: "POST", body: JSON.stringify(body) }, false);
    setSession(payload);
    setModal(false);
    showToast(mode === "register" ? "Your Onco Aegis AI account is ready." : "Welcome back.");
  } catch (error) {
    message.textContent = errorMessage(error);
  } finally {
    button.disabled = false;
    button.querySelector("span:first-child").textContent = mode === "register" ? "Create account" : "Sign in";
  }
}

async function signOut() {
  try {
    if (state.accessToken && state.refreshToken) await api("/auth/logout", { method: "POST", body: JSON.stringify({ refresh_token: state.refreshToken }) }, false);
  } catch { /* Local cleanup still completes when the server session is gone. */ }
  clearSession();
  showToast("You have been signed out.");
}

async function loadGuideStatus() {
  try {
    const status = await api("/information/cancer/status", {}, false);
    setServiceStatus(status.configured_ai ? "AI guide ready" : "Educational guide ready", "green");
  } catch {
    setServiceStatus("Guide status unavailable", "amber");
  }
}

async function restoreSession() {
  if (!state.accessToken || !state.refreshToken) return;
  try {
    state.user = await api("/auth/me", {}, false);
    renderUser();
  } catch {
    clearSession();
  }
}

function setActiveNavigation(action) {
  $$('[data-nav-action]').forEach((button) => {
    button.classList.toggle("is-active", button.dataset.navAction === action);
  });
}

function closeMobileNavigation() {
  const toggle = $("#mobile-nav-toggle");
  const nav = $("#primary-nav");
  if (!toggle || !nav) return;
  toggle.classList.remove("is-open");
  toggle.setAttribute("aria-expanded", "false");
  nav.classList.remove("is-open");
}

function showHistoryPanel() {
  if (!state.user) {
    setModal(true);
    showToast("Sign in to view and continue your saved conversations.");
    return;
  }
  $("#history-panel").hidden = false;
  loadHistory();
}

function scrollToSection(selector) {
  const target = $(selector);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function handleNavigation(action) {
  closeMobileNavigation();
  if (action === "home") {
    if (state.user) showWorkspace();
    else window.scrollTo({ top: 0, behavior: "smooth" });
  } else if (action === "guide") {
    if (!state.user) {
      setModal(true);
      return;
    }
    showWorkspace();
    window.setTimeout(() => scrollToSection("#guide-panel"), 60);
  } else if (action === "history") {
    showHistoryPanel();
  } else if (action === "resources") {
    scrollToSection("#resources");
  } else if (action === "about") {
    scrollToSection("#about");
  }
  setActiveNavigation(action);
}

function applyQuickExample(button) {
  if (!state.user) {
    setModal(true);
    return;
  }
  const topic = button.dataset.exampleTopic || "";
  const prompt = button.dataset.examplePrompt || "";
  showWorkspace();
  $("#cancer-topic").value = topic;
  $("#question-text").value = prompt;
  updateCharacterCount();
  window.setTimeout(() => {
    scrollToSection("#guide-panel");
    $("#question-text").focus();
  }, 70);
}

function configureChatDropZones() {
  $$(".chat-attach").forEach((zone) => {
    const input = zone.querySelector("input");
    if (!input) return;
    zone.addEventListener("dragover", (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-dragging"));
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
      const files = event.dataTransfer.files;
      if (input.id === "chat-files") setChatFiles(files);
      if (input.id === "chat-document") {
        state.chatDocument = files?.[0] || null;
        renderChatDocument();
      }
    });
  });
}

function start() {
  $("#question-form").addEventListener("submit", submitQuestion);
  $("#question-text").addEventListener("input", updateCharacterCount);
  $("#chat-files").addEventListener("change", (event) => setChatFiles(event.target.files));
  $("#chat-document").addEventListener("change", (event) => { state.chatDocument = event.target.files?.[0] || null; renderChatDocument(); });
  $("#chat-model").addEventListener("change", (event) => selectCoverage(event.target.value));
  $$('[data-coverage-model]').forEach((button) => button.addEventListener("click", () => selectCoverage(button.dataset.coverageModel)));
  $$(".prompt-chip").forEach((button) => button.addEventListener("click", () => selectPrompt(button)));
  $$("[data-language]").forEach((button) => button.addEventListener("click", () => selectLanguage(button.dataset.language)));
  $("#ask-another").addEventListener("click", () => { $("#question-text").focus(); window.scrollTo({ top: $("#assistant-title").getBoundingClientRect().top + window.scrollY - 110, behavior: "smooth" }); });
  $("#back-workspace").addEventListener("click", showWorkspace);
  $("#open-auth").addEventListener("click", () => setModal(true));
  $("#gateway-signin").addEventListener("click", () => setModal(true));
  $("#refresh-workspace").addEventListener("click", () => window.location.reload());
  $("#open-history").addEventListener("click", showHistoryPanel);
  $("#close-history").addEventListener("click", () => { $("#history-panel").hidden = true; });
  $("#new-chat").addEventListener("click", startNewChat);
  $("#delete-selected-history").addEventListener("click", deleteSelectedHistory);
  $("#close-auth").addEventListener("click", () => setModal(false));
  $("#auth-modal").addEventListener("click", (event) => { if (event.target === event.currentTarget) setModal(false); });
  $("#user-button").addEventListener("click", signOut);
  $("#auth-form").addEventListener("submit", submitAuth);
  $("#analysis-form").addEventListener("submit", submitAnalysis);
  $("#analysis-model").addEventListener("change", updateAnalysisModel);
  $("#analysis-files").addEventListener("change", (event) => setAnalysisFiles(event.target.files));
  $("#upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); $("#upload-zone").classList.add("is-dragging"); });
  $("#upload-zone").addEventListener("dragleave", () => $("#upload-zone").classList.remove("is-dragging"));
  $("#upload-zone").addEventListener("drop", (event) => {
    event.preventDefault();
    $("#upload-zone").classList.remove("is-dragging");
    setAnalysisFiles(event.dataTransfer.files);
  });
  $("#mobile-nav-toggle").addEventListener("click", () => {
    const toggle = $("#mobile-nav-toggle");
    const nav = $("#primary-nav");
    const willOpen = !nav.classList.contains("is-open");
    toggle.classList.toggle("is-open", willOpen);
    toggle.setAttribute("aria-expanded", String(willOpen));
    nav.classList.toggle("is-open", willOpen);
  });
  $$('[data-nav-action]').forEach((button) => button.addEventListener("click", () => handleNavigation(button.dataset.navAction)));
  $$(".rail-example").forEach((button) => button.addEventListener("click", () => applyQuickExample(button)));
  $$('[data-auth-mode]').forEach((button) => button.addEventListener("click", () => setAuthMode(button.dataset.authMode)));
  setAuthMode("login");
  updateCharacterCount();
  renderChatFiles();
  $("#upload-guidance").textContent = "Choose an image route to see the best supported file type.";
  renderHistoryList();
  updateAnalysisModel();
  renderAnalysisFiles();
  configureChatDropZones();
  setActiveNavigation("home");
  renderUser();
  loadGuideStatus();
  loadProviders();
  restoreSession();
}

document.addEventListener("DOMContentLoaded", start);
