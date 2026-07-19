"""Bootstraps the local llm_engine + Bedrock backend stack for Berky.

Provisions (idempotently, reusing anything already running) MongoDB,
ChromaDB, and the llm_engine Node service on whatever host machine this
Reachy Mini app is running on, seeds the Berky operator account/topic/
conversation via llm_engine's REST API, and writes the resulting
conversation ID/credentials into berkie_reachy's own config so
``BerkyLiveHandler`` picks it up automatically - all without any manual
terminal commands.
"""

from __future__ import annotations

import os
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from berkie_reachy.llm_engine_bootstrap import state, repo, mongo, chroma, node, seed

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]  # (step_name, message)


@dataclass
class StackStatus:
    """Snapshot of bootstrap progress, polled by the settings UI."""

    node_found: bool = False
    node_supported: bool = False
    yarn_ready: bool = False
    mongo_running: bool = False
    chroma_running: bool = False
    chroma_installed: bool = False
    llm_engine_healthy: bool = False
    seeded: bool = False
    error: Optional[str] = None
    needs_action: Optional[str] = None  # human-readable instructions, if blocked
    skipped: bool = False
    done: bool = False


@dataclass
class StackResult:
    """What main.py needs after a successful (or skipped) bootstrap."""

    conversation_id: Optional[str]
    username: Optional[str]
    password: Optional[str]
    skipped: bool = False


class _Registry:
    """Shared state between the background bootstrap loop and the settings UI."""

    def __init__(self) -> None:
        self.skip_requested = threading.Event()
        self.done = threading.Event()
        self.status = StackStatus()
        self.lock = threading.Lock()
        self.bedrock_api_key = ""
        self.bedrock_base_url = ""
        self.openai_api_key = ""
        self.tavily_api_key = ""
        self.result: Optional["StackResult"] = None

    def set_bedrock_credentials(self, api_key: str, base_url: str) -> None:
        with self.lock:
            self.bedrock_api_key = api_key
            self.bedrock_base_url = base_url

    def set_openai_api_key(self, api_key: str) -> None:
        with self.lock:
            self.openai_api_key = api_key

    def set_tavily_api_key(self, api_key: str) -> None:
        with self.lock:
            self.tavily_api_key = api_key


def _read_persisted_config(instance_path: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Read any previously persisted conversation ID/username/password from the instance .env."""
    if not instance_path:
        return None, None, None
    env_path = Path(instance_path) / ".env"
    if not env_path.exists():
        return None, None, None
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip()
    return (
        values.get("BERKIE_LLM_ENGINE_CONVERSATION_ID") or None,
        values.get("BERKIE_LLM_ENGINE_USERNAME") or None,
        values.get("BERKIE_LLM_ENGINE_PASSWORD") or None,
    )


def _persist_config(instance_path: Optional[str], updates: dict[str, str]) -> None:
    """Write updates into the instance .env (create/replace lines) and mutate live config + os.environ.

    Mirrors console.py's _persist_api_key pattern for OPENAI_API_KEY.
    """
    from berkie_reachy.config import config

    for key, value in updates.items():
        os.environ[key] = value
        try:
            setattr(config, key, value)
        except Exception:
            pass

    if not instance_path:
        return
    inst = Path(instance_path)
    inst.mkdir(parents=True, exist_ok=True)
    env_path = inst / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    for key, value in updates.items():
        replaced = False
        for i, ln in enumerate(lines):
            if ln.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_openai_key(explicit: str = "") -> str:
    """Resolve an OpenAI-compatible key for embeddings.

    Prefers an explicitly-supplied value (e.g. entered in the Bedrock settings
    panel), then falls back to berkie_reachy's own OPENAI_API_KEY (already
    collected via console.py's first-run flow, when that flow is in use).
    """
    if explicit:
        return explicit
    from berkie_reachy.config import config as berkie_config

    return getattr(berkie_config, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", "")


def _resolve_tavily_key(explicit: str = "") -> str:
    """Resolve a Tavily API key for the web_search tool, if configured.

    Optional - unlike Bedrock/OpenAI, missing this doesn't block the bootstrap;
    llm_engine's web_search tool just fails gracefully per-call without it.
    """
    if explicit:
        return explicit
    from berkie_reachy.config import config as berkie_config

    return getattr(berkie_config, "TAVILY_API_KEY", None) or os.getenv("TAVILY_API_KEY", "")


def _build_llm_engine_env(
    bedrock_api_key: str, bedrock_base_url: str, openai_api_key: str, tavily_api_key: str = ""
) -> dict[str, str]:
    """Build the env for the llm_engine child process.

    Bedrock credentials cover chat completions. Embeddings (used for RAG -
    topic/transcript vector storage, hit as soon as a topic is created) are a
    *separate* OpenAI-specific dependency (DEFAULT_OPENAI_API_KEY /
    DEFAULT_EMBEDDINGS_API_KEY in llm_engine's config) - found the hard way
    when topic creation 500'd with "Missing credentials" during testing.
    """
    env = dict(os.environ)
    env.update(
        {
            "NODE_ENV": "production",
            "MONGODB_URL": mongo.MONGODB_URL,
            "JWT_SECRET": state.get_or_create_jwt_secret(),
            "PORT": str(state.LLM_ENGINE_PORT),
            "WEBSOCKET_BASE_PORT": str(state.LLM_ENGINE_WS_PORT),
            "CHROMA_DB_URL": chroma.CHROMA_DB_URL,
            "BEDROCK_API_KEY": bedrock_api_key,
            "BEDROCK_BASE_URL": bedrock_base_url,
        }
    )
    if openai_api_key:
        env["DEFAULT_OPENAI_API_KEY"] = openai_api_key
    if tavily_api_key:
        env["TAVILY_API_KEY"] = tavily_api_key
    return env


def run_bootstrap(
    *,
    instance_path: Optional[str] = None,
    bedrock_api_key: str = "",
    bedrock_base_url: str = "",
    openai_api_key: str = "",
    tavily_api_key: str = "",
    registry: Optional[_Registry] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> StackResult:
    """Run the full bootstrap synchronously. Safe to call repeatedly (idempotent)."""
    registry = registry or _Registry()

    def report(step: str, message: str) -> None:
        logger.info("[llm_engine_bootstrap] %s: %s", step, message)
        if on_progress:
            on_progress(step, message)

    state.ensure_dirs()

    with state.BootstrapLock():
        detection = node.detect_node_and_yarn()
        registry.status.node_found = detection.node_found
        registry.status.node_supported = detection.node_supported
        if detection.instructions:
            registry.status.needs_action = detection.instructions
            report("node", detection.instructions)
            return StackResult(None, None, None, skipped=True)
        registry.status.yarn_ready = True
        report("node", f"Node {detection.node_version} + Yarn OK")

        if not bedrock_api_key or not bedrock_base_url:
            msg = "BEDROCK_API_KEY/BEDROCK_BASE_URL not configured yet - waiting for operator input."
            registry.status.needs_action = msg
            report("bedrock", msg)
            return StackResult(None, None, None, skipped=True)

        # Embeddings (used for RAG - topic/transcript vector storage, hit as
        # soon as a topic is created) need their own OpenAI-compatible key,
        # separate from Bedrock. Check for one *before* attempting to seed,
        # rather than letting llm_engine 500 with a generic "Internal Server
        # Error" that hides the real "Missing credentials" cause (found the
        # hard way - the actual error only shows up in llm_engine's own log).
        resolved_openai_key = _resolve_openai_key(openai_api_key)
        if not resolved_openai_key:
            msg = (
                "OpenAI API key not configured yet - needed for embeddings (RAG topic/transcript "
                "vector storage), separate from the Bedrock credentials above. Enter one on the "
                "settings page to continue."
            )
            registry.status.needs_action = msg
            report("openai_embeddings", msg)
            return StackResult(None, None, None, skipped=True)

        src_dir = repo.ensure_llm_engine_source()
        report("repo", f"llm_engine source ready at {src_dir}")

        try:
            mongo.ensure_mongo_running()
            registry.status.mongo_running = True
            report("mongo", "MongoDB running")
        except mongo.MongoNotAvailableError as e:
            registry.status.needs_action = str(e)
            report("mongo", str(e))
            return StackResult(None, None, None, skipped=True)

        registry.status.chroma_installed = chroma.is_chromadb_installed()
        if not registry.status.chroma_installed:
            # Safe to fully automate: this installs into Chroma's own isolated
            # venv, not berkie_reachy's/reachy_mini's shared environment, so
            # it can't conflict with reachy_mini's pinned dependencies the
            # way installing it into the shared env did (see chroma.py).
            report("chroma", "Installing chromadb (first run on this host)...")
            try:
                chroma.install_chromadb()
                registry.status.chroma_installed = True
            except Exception as e:
                msg = f"Failed to install chromadb: {e}"
                registry.status.needs_action = msg
                report("chroma", msg)
                return StackResult(None, None, None, skipped=True)
        chroma.ensure_chroma_running()
        registry.status.chroma_running = True
        report("chroma", "Chroma running")

        yarn_cmd = node.ensure_yarn_ready()
        node.ensure_dependencies_installed(src_dir, yarn_cmd)
        node.ensure_built(src_dir, yarn_cmd)
        resolved_tavily_key = _resolve_tavily_key(tavily_api_key)
        env = _build_llm_engine_env(bedrock_api_key, bedrock_base_url, resolved_openai_key, resolved_tavily_key)
        node.ensure_llm_engine_running(src_dir, env)
        registry.status.llm_engine_healthy = True
        report("llm_engine", "llm_engine healthy")

        existing_id, existing_user, existing_pass = _read_persisted_config(instance_path)
        existing = None
        if existing_id and existing_user and existing_pass:
            existing = seed.SeedResult(conversation_id=existing_id, username=existing_user, password=existing_pass)

        def persist_credentials(username: str, password: str) -> None:
            _persist_config(
                instance_path,
                {"BERKIE_LLM_ENGINE_USERNAME": username, "BERKIE_LLM_ENGINE_PASSWORD": password},
            )

        result = seed.ensure_seeded(
            node.LLM_ENGINE_URL,
            existing=existing,
            persist_credentials=persist_credentials,
        )
        _persist_config(
            instance_path,
            {
                "BERKIE_LLM_ENGINE_CONVERSATION_ID": result.conversation_id,
                "BERKIE_LLM_ENGINE_USERNAME": result.username,
                "BERKIE_LLM_ENGINE_PASSWORD": result.password,
                "BERKIE_LLM_ENGINE_BASE_URL": f"{node.LLM_ENGINE_URL}/v1",
                "BERKY_LLM_ENGINE_SOCKET_URL": f"http://127.0.0.1:{state.LLM_ENGINE_WS_PORT}",
            },
        )
        registry.status.seeded = True
        registry.status.done = True
        report("seed", f"Berky conversation ready: {result.conversation_id}")

        return StackResult(
            conversation_id=result.conversation_id,
            username=result.username,
            password=result.password,
        )


def stop_stack() -> None:
    """Stop every service this process started (safe to call even if nothing was started)."""
    node.stop_llm_engine_if_started_by_us()
    chroma.stop_chroma_if_started_by_us()
    mongo.stop_mongo_if_started_by_us()


def ensure_llm_engine_stack(
    *,
    settings_app=None,
    instance_path: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
    bedrock_api_key: str = "",
    bedrock_base_url: str = "",
    openai_api_key: str = "",
    tavily_api_key: str = "",
    poll_interval: float = 5.0,
    # A genuinely fresh install (clone + yarn install + build + chromadb's
    # own fairly large dependency tree) can take a few minutes end to end;
    # 120s was too short and could time out into the OpenAI-only fallback
    # even when the bootstrap would have succeeded moments later. Subsequent
    # launches are fast (everything's cached/already healthy), so this only
    # matters for the very first run on a given host.
    wait_timeout: Optional[float] = 300.0,
) -> StackResult:
    """Entry point called from main.py's run(), before the use_berky_backend check.

    Mounts settings UI routes (if settings_app given), registers a shutdown
    watcher, then retries the bootstrap in the background - mirroring
    console.py's existing block/poll pattern for OPENAI_API_KEY - until it
    succeeds, the operator hits "skip" via the settings UI, or wait_timeout
    elapses. This call blocks for at most wait_timeout seconds; if the stack
    isn't ready by then, it returns a "skipped" result so the robot still
    starts (falling back to the existing OpenAI-only path) rather than
    hanging indefinitely on a missing dependency the operator hasn't
    resolved yet. The background retry loop keeps running past that point,
    so a later settings-UI update can still complete the bootstrap - the
    caller just doesn't wait around for it.
    """
    registry = _Registry()
    registry.bedrock_api_key = bedrock_api_key
    registry.bedrock_base_url = bedrock_base_url
    registry.openai_api_key = openai_api_key
    registry.tavily_api_key = tavily_api_key

    if settings_app is not None:
        try:
            from berkie_reachy.llm_engine_bootstrap.settings_ui import mount_routes

            mount_routes(settings_app, registry, instance_path=instance_path)
        except Exception:
            logger.exception("Failed to mount llm_engine_bootstrap settings UI routes")

    if stop_event is not None:
        def _watch_stop() -> None:
            stop_event.wait()
            stop_stack()

        threading.Thread(target=_watch_stop, daemon=True).start()

    def _retry_loop() -> None:
        while not registry.skip_requested.is_set() and not registry.done.is_set():
            with registry.lock:
                api_key = registry.bedrock_api_key
                base_url = registry.bedrock_base_url
                openai_key = registry.openai_api_key
                tavily_key = registry.tavily_api_key
            try:
                result = run_bootstrap(
                    instance_path=instance_path,
                    bedrock_api_key=api_key,
                    bedrock_base_url=base_url,
                    openai_api_key=openai_key,
                    tavily_api_key=tavily_key,
                    registry=registry,
                )
                if not result.skipped:
                    registry.result = result
                    registry.done.set()
                    return
            except Exception as e:
                registry.status.error = str(e)
                logger.exception("llm_engine bootstrap attempt failed; will retry")
            registry.done.wait(poll_interval)

    thread = threading.Thread(target=_retry_loop, daemon=True)
    thread.start()

    finished = registry.done.wait(wait_timeout)
    if finished and registry.result is not None:
        return registry.result
    if registry.skip_requested.is_set():
        return StackResult(None, None, None, skipped=True)
    logger.info(
        "llm_engine bootstrap not ready after %ss; continuing without it for now "
        "(it keeps retrying in the background; the settings UI can supply missing "
        "credentials/dependencies at any time).",
        wait_timeout,
    )
    return StackResult(None, None, None, skipped=True)
