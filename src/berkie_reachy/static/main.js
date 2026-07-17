const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchWithTimeout(url, options = {}, timeoutMs = 2000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function waitForStatus(timeoutMs = 15000) {
  const loadingText = document.querySelector("#loading p");
  let attempts = 0;
  const deadline = Date.now() + timeoutMs;
  while (true) {
    attempts += 1;
    try {
      const url = new URL("/status", window.location.origin);
      url.searchParams.set("_", Date.now().toString());
      const resp = await fetchWithTimeout(url, {}, 2000);
      if (resp.ok) return await resp.json();
    } catch (e) {}
    if (loadingText) {
      loadingText.textContent = attempts > 8 ? "Starting backend…" : "Loading…";
    }
    if (Date.now() >= deadline) return null;
    await sleep(500);
  }
}

async function validateKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/validate_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.error || "validation_failed");
  }
  return data;
}

async function saveKey(key) {
  const body = { openai_api_key: key };
  const resp = await fetch("/openai_api_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

async function init() {
  const loading = document.getElementById("loading");
  const statusEl = document.getElementById("status");
  const formPanel = document.getElementById("form-panel");
  const configuredPanel = document.getElementById("configured");
  const saveBtn = document.getElementById("save-btn");
  const changeKeyBtn = document.getElementById("change-key-btn");
  const input = document.getElementById("api-key");

  show(loading, true);
  show(formPanel, false);
  show(configuredPanel, false);

  const st = (await waitForStatus()) || { has_key: false, needs_openai_key: true };

  // needs_openai_key is false when the active handler (e.g. the Bedrock/
  // llm_engine backend) doesn't use OpenAI at all - showing this panel in
  // that case just invites confusion over a credential that isn't needed.
  if (st.needs_openai_key === false) {
    show(formPanel, false);
    show(configuredPanel, false);
  } else if (st.has_key) {
    show(configuredPanel, true);
  } else {
    show(formPanel, true);
  }
  show(loading, false);

  changeKeyBtn.addEventListener("click", () => {
    show(configuredPanel, false);
    show(formPanel, true);
    input.value = "";
    statusEl.textContent = "";
    statusEl.className = "status";
  });

  input.addEventListener("input", () => {
    input.classList.remove("error");
  });

  saveBtn.addEventListener("click", async () => {
    const key = input.value.trim();
    if (!key) {
      statusEl.textContent = "Please enter a valid key.";
      statusEl.className = "status warn";
      input.classList.add("error");
      return;
    }
    statusEl.textContent = "Validating API key...";
    statusEl.className = "status";
    input.classList.remove("error");
    try {
      const validation = await validateKey(key);
      if (!validation.valid) {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
        statusEl.className = "status error";
        input.classList.add("error");
        return;
      }
      statusEl.textContent = "Key valid! Saving...";
      statusEl.className = "status ok";
      await saveKey(key);
      statusEl.textContent = "Saved. Reloading…";
      statusEl.className = "status ok";
      window.location.reload();
    } catch (e) {
      input.classList.add("error");
      if (e.message === "invalid_api_key") {
        statusEl.textContent = "Invalid API key. Please check your key and try again.";
      } else {
        statusEl.textContent = "Failed to validate/save key. Please try again.";
      }
      statusEl.className = "status error";
    }
  });
}

const LLM_BACKEND_STEPS = [
  { key: "node_found", label: "Node.js found" },
  { key: "yarn_ready", label: "Yarn ready" },
  { key: "mongo_running", label: "MongoDB running" },
  { key: "chroma_running", label: "ChromaDB running" },
  { key: "llm_engine_healthy", label: "llm_engine running" },
  { key: "seeded", label: "Berky conversation ready" },
];

async function fetchLlmBackendStatus() {
  try {
    const resp = await fetchWithTimeout("/llm_backend/status", {}, 2000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function saveBedrockCredentials(apiKey, baseUrl, openaiKey) {
  const resp = await fetch("/llm_backend/bedrock_credentials", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bedrock_api_key: apiKey, bedrock_base_url: baseUrl, openai_api_key: openaiKey }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

async function skipLlmBackend() {
  const resp = await fetch("/llm_backend/skip", { method: "POST" });
  if (!resp.ok) throw new Error("skip_failed");
  return await resp.json();
}

function renderLlmBackendChecklist(listEl, status) {
  listEl.innerHTML = "";
  for (const step of LLM_BACKEND_STEPS) {
    const li = document.createElement("li");
    const done = !!status[step.key];
    li.className = done ? "done" : "";
    const dot = document.createElement("span");
    dot.className = "dot";
    li.appendChild(dot);
    li.appendChild(document.createTextNode(step.label));
    listEl.appendChild(li);
  }
}

async function initLlmBackendPanel() {
  const panel = document.getElementById("llm-backend-panel");
  const chip = document.getElementById("llm-backend-chip");
  const checklist = document.getElementById("llm-backend-checklist");
  const needsEl = document.getElementById("llm-backend-needs");
  const form = document.getElementById("llm-backend-form");
  const apiKeyInput = document.getElementById("bedrock-api-key");
  const baseUrlInput = document.getElementById("bedrock-base-url");
  const openaiKeyInput = document.getElementById("llm-backend-openai-key");
  const saveBtn = document.getElementById("llm-backend-save-btn");
  const skipBtn = document.getElementById("llm-backend-skip-btn");
  const statusEl = document.getElementById("llm-backend-status");

  // Only show this panel at all if the llm_backend routes are actually mounted
  // (they aren't when the app is running in plain OpenAI-only mode). Retry
  // for a while rather than giving up after one check - these routes mount
  // as soon as main.py's run() reaches the bootstrap call, but that can be
  // a few seconds into startup (robot connection, vision setup, etc. happen
  // first), and a single failed check here used to permanently hide the
  // panel for the rest of the page's lifetime (only a full reload, e.g. the
  // one triggered by saving an OpenAI key elsewhere on this page, gave it
  // another chance - which looked like "the panel appears after entering an
  // OpenAI key" but was really just incidental timing).
  let initialStatus = null;
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    initialStatus = await fetchLlmBackendStatus();
    if (initialStatus) break;
    await sleep(1000);
  }
  if (!initialStatus) return;

  show(panel, true);

  let polling = true;
  const poll = async () => {
    while (polling) {
      const status = await fetchLlmBackendStatus();
      if (status) {
        renderLlmBackendChecklist(checklist, status);
        if (status.done) {
          chip.textContent = "Ready";
          chip.className = "chip chip-ok";
          show(form, false);
          show(needsEl, false);
          polling = false;
          break;
        } else if (status.skipped) {
          chip.textContent = "Skipped";
          show(form, false);
          show(needsEl, false);
          polling = false;
          break;
        } else {
          chip.textContent = "Setting up";
          const needs = await fetch("/llm_backend/needs")
            .then((r) => r.json())
            .catch(() => ({}));
          if (needs.instructions) {
            needsEl.textContent = needs.instructions;
            show(needsEl, true);
          } else {
            show(needsEl, false);
          }
          show(form, true);
        }
      }
      await sleep(3000);
    }
  };
  poll();

  saveBtn.addEventListener("click", async () => {
    const apiKey = apiKeyInput.value.trim();
    const baseUrl = baseUrlInput.value.trim();
    const openaiKey = openaiKeyInput.value.trim();
    if (!apiKey || !baseUrl) {
      statusEl.textContent = "Enter both the Bedrock API key and base URL.";
      statusEl.className = "status warn";
      return;
    }
    // openaiKey is optional here - it may already be configured (e.g. via the
    // legacy OpenAI panel or a previous save). The backend will report a
    // clear needs_action message if it's actually still missing.
    statusEl.textContent = "Saving...";
    statusEl.className = "status";
    try {
      await saveBedrockCredentials(apiKey, baseUrl, openaiKey);
      statusEl.textContent = "Saved. Trying to connect...";
      statusEl.className = "status ok";
    } catch (e) {
      statusEl.textContent = "Failed to save credentials. Please try again.";
      statusEl.className = "status error";
    }
  });

  skipBtn.addEventListener("click", async () => {
    statusEl.textContent = "Skipping local backend...";
    statusEl.className = "status";
    try {
      await skipLlmBackend();
      statusEl.textContent = "Skipped. Using plain OpenAI conversation mode.";
      statusEl.className = "status ok";
    } catch (e) {
      statusEl.textContent = "Failed to skip. Please try again.";
      statusEl.className = "status error";
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  init();
  initLlmBackendPanel();
});