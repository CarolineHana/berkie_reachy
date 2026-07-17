"""FastAPI routes for the llm_engine backend bootstrap, mounted onto the same
settings_app instance console.py extends for the OpenAI-key flow.

Routes live under /llm_backend/* so they never collide with console.py's
existing /, /static, /status, /ready, /openai_api_key, /validate_api_key.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    BaseModel = object  # type: ignore


class BedrockCredentials(BaseModel):
    """Payload for POST /llm_backend/bedrock_credentials.

    openai_api_key is separate from Bedrock: embeddings (RAG topic/transcript
    vector storage, hit as soon as a topic is created) need their own
    OpenAI-compatible key regardless of which platform handles chat. Optional
    here since it may already be set via the legacy OpenAI panel/config.
    """

    bedrock_api_key: str
    bedrock_base_url: str
    openai_api_key: str = ""


def mount_routes(app: "FastAPI", registry, instance_path=None) -> None:
    """Attach the bootstrap status/control endpoints to ``app``."""
    from berkie_reachy.llm_engine_bootstrap import chroma

    @app.get("/llm_backend/status")
    def _status() -> dict:
        s = registry.status
        return {
            "node_found": s.node_found,
            "node_supported": s.node_supported,
            "yarn_ready": s.yarn_ready,
            "mongo_running": s.mongo_running,
            "chroma_installed": s.chroma_installed,
            "chroma_running": s.chroma_running,
            "llm_engine_healthy": s.llm_engine_healthy,
            "seeded": s.seeded,
            "done": registry.done.is_set(),
            "skipped": registry.skip_requested.is_set(),
            "error": s.error,
        }

    @app.get("/llm_backend/needs")
    def _needs() -> dict:
        s = registry.status
        return {"instructions": s.needs_action}

    @app.post("/llm_backend/bedrock_credentials")
    def _set_bedrock_credentials(payload: BedrockCredentials) -> dict:
        # Deferred import: settings_ui is imported from __init__.py, so a
        # module-level import back would be circular.
        from berkie_reachy.llm_engine_bootstrap import _persist_config

        api_key = payload.bedrock_api_key.strip()
        base_url = payload.bedrock_base_url.strip()
        openai_key = payload.openai_api_key.strip()
        registry.set_bedrock_credentials(api_key, base_url)
        # Persist immediately so the operator doesn't have to re-enter these
        # every launch (mirrors console.py's OPENAI_API_KEY persistence).
        updates = {"BEDROCK_API_KEY": api_key, "BEDROCK_BASE_URL": base_url}
        if openai_key:
            registry.set_openai_api_key(openai_key)
            # Same config attribute console.py's OPENAI_API_KEY flow uses, so
            # either panel can supply it and both consumers see the same value.
            updates["OPENAI_API_KEY"] = openai_key
        _persist_config(instance_path, updates)
        # Clear any stale "waiting for credentials" message and un-skip so the
        # background retry loop picks the new values up on its next attempt.
        registry.status.needs_action = None
        return {"status": "ok"}

    @app.post("/llm_backend/skip")
    def _skip() -> dict:
        registry.skip_requested.set()
        registry.status.skipped = True
        logger.info("Operator skipped the local llm_engine backend setup; using OpenAI-only mode.")
        return {"status": "skipped"}

    @app.post("/llm_backend/install_chromadb")
    def _install_chromadb() -> dict:
        def _do_install() -> None:
            try:
                chroma.install_chromadb()
                registry.status.chroma_installed = True
                registry.status.needs_action = None
            except Exception as e:
                registry.status.error = f"Failed to install chromadb: {e}"

        threading.Thread(target=_do_install, daemon=True).start()
        return {"status": "installing"}
