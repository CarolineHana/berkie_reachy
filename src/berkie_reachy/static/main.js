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
  initInteractionModePanel();
});