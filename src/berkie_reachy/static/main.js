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

function show(el, flag) {
  el.classList.toggle("hidden", !flag);
}

async function fetchBerkyConnectionStatus() {
  try {
    const resp = await fetchWithTimeout("/berky_connection/status", {}, 2000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function saveBerkyConnection(conversationId, username, password, passcode) {
  const resp = await fetch("/berky_connection/credentials", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      username: username,
      password: password,
      passcode: passcode,
    }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "save_failed");
  }
  return await resp.json();
}

async function initBerkyConnectionPanel() {
  const panel = document.getElementById("berky-connection-panel");
  const chip = document.getElementById("berky-connection-chip");
  const conversationIdInput = document.getElementById("berky-conversation-id");
  const usernameInput = document.getElementById("berky-username");
  const passwordInput = document.getElementById("berky-password");
  const passcodeInput = document.getElementById("berky-passcode");
  const saveBtn = document.getElementById("berky-connection-save-btn");
  const statusEl = document.getElementById("berky-connection-status");

  // Same retry-for-a-while pattern as the other panels - these routes mount a few
  // seconds into startup, not immediately on page load.
  let initialStatus = null;
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    initialStatus = await fetchBerkyConnectionStatus();
    if (initialStatus) break;
    await sleep(1000);
  }
  if (!initialStatus) return;

  show(panel, true);

  const render = (status) => {
    const complete = status.conversation_id_set && status.username_set && status.password_set;
    if (status.active_this_session) {
      chip.textContent = "Connected";
      chip.className = "chip chip-ok";
    } else if (complete) {
      chip.textContent = "Saved - restart to apply";
      chip.className = "chip";
    } else {
      chip.textContent = "Not connected";
      chip.className = "chip";
    }
    conversationIdInput.placeholder = status.conversation_id_set ? "(already set)" : "Mongo ObjectId";
    usernameInput.placeholder = status.username_set ? "(already set)" : "berky-operator-...";
    passwordInput.placeholder = status.password_set ? "(already set)" : "Operator account password";
    passcodeInput.placeholder = status.passcode_set ? "(already set)" : "Optional";
  };
  render(initialStatus);

  saveBtn.addEventListener("click", async () => {
    const conversationId = conversationIdInput.value.trim();
    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();
    const passcode = passcodeInput.value.trim();
    if (!conversationId || !username || !password) {
      statusEl.textContent = "Conversation ID, username, and password are all required.";
      statusEl.className = "status warn";
      return;
    }
    statusEl.textContent = "Saving...";
    statusEl.className = "status";
    try {
      await saveBerkyConnection(conversationId, username, password, passcode);
      const status = await fetchBerkyConnectionStatus();
      if (status) render(status);
      conversationIdInput.value = "";
      usernameInput.value = "";
      passwordInput.value = "";
      passcodeInput.value = "";
      statusEl.textContent = "Saved. Restart the app to connect with these credentials.";
      statusEl.className = "status ok";
    } catch (e) {
      statusEl.textContent = "Failed to save credentials. Please try again.";
      statusEl.className = "status error";
    }
  });
}

async function fetchInteractionMode() {
  try {
    const resp = await fetchWithTimeout("/interaction_mode", {}, 2000);
    if (!resp.ok) return null;
    return await resp.json();
  } catch (e) {
    return null;
  }
}

async function setInteractionMode(mode) {
  const resp = await fetch("/interaction_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || "set_mode_failed");
  }
  return await resp.json();
}

async function initInteractionModePanel() {
  const panel = document.getElementById("interaction-mode-panel");
  const chip = document.getElementById("interaction-mode-chip");
  const communityBtn = document.getElementById("mode-community-btn");
  const welcomerBtn = document.getElementById("mode-welcomer-btn");
  const statusEl = document.getElementById("interaction-mode-status");

  // The /interaction_mode route only exists once console.py's settings UI has
  // initialized, which happens a few seconds into startup (robot connection,
  // vision setup, etc. happen first) - a single check right on page load can
  // land before that, so retry for a while rather than giving up.
  let initial = null;
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    initial = await fetchInteractionMode();
    if (initial) break;
    await sleep(1000);
  }
  if (!initial || !initial.available) return;
  show(panel, true);

  const render = (mode) => {
    const isWelcomer = mode === "welcomer";
    chip.textContent = isWelcomer ? "Welcomer" : "Community Assistant";
    chip.className = "chip chip-ok";
    communityBtn.classList.toggle("ghost", isWelcomer);
    welcomerBtn.classList.toggle("ghost", !isWelcomer);
  };
  render(initial.mode);

  const switchTo = async (mode) => {
    statusEl.textContent = "Switching...";
    statusEl.className = "status";
    try {
      const result = await setInteractionMode(mode);
      render(result.mode);
      statusEl.textContent = "Switched.";
      statusEl.className = "status ok";
    } catch (e) {
      statusEl.textContent = "Failed to switch mode. Please try again.";
      statusEl.className = "status error";
    }
  };

  communityBtn.addEventListener("click", () => switchTo("community_assistant"));
  welcomerBtn.addEventListener("click", () => switchTo("welcomer"));
}

window.addEventListener("DOMContentLoaded", () => {
  initBerkyConnectionPanel();
  initInteractionModePanel();
});