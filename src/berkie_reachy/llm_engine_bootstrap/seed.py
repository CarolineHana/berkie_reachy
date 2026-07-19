"""Idempotently seed the Berky operator account, topic, and conversation via llm_engine's REST API.

Deliberately goes through the REST API rather than writing MongoDB documents
directly: POST /v1/conversations's agentTypes handling runs through Mongoose's
pre('validate') hook (agent.deepPatch(...); agent.save()), which is what
correctly populates llmTemplates defaults. A raw DB write bypassing that hook
is exactly what caused a stale-llmTemplates bug found and hand-fixed earlier
this session - seeding through the API avoids reintroducing it.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_AGENT_CONFIG = {
    "botName": "Berkie",
    "personality": "sarcastic-expert",
    "tools": ["web_search"],
    "recentTranscriptTurns": 20,
}
DEFAULT_LLM_PLATFORM = "bedrock"
DEFAULT_LLM_MODEL = "us.anthropic.claude-opus-4-6-v1"


class SeedError(RuntimeError):
    """Raised when seeding the Berky operator account/conversation fails."""


@dataclass
class SeedResult:
    """Everything berkie_reachy's config needs after a successful seed."""

    conversation_id: str
    username: str
    password: str


def _register_operator(client: httpx.Client) -> tuple[str, str, str]:
    """Register a fresh operator account. Returns (username, password, access_token)."""
    pseudo_resp = client.get("/v1/auth/newPseudonym")
    pseudo_resp.raise_for_status()
    pseudo_data = pseudo_resp.json()

    username = f"berky-operator-{secrets.token_hex(4)}"
    # llm_engine requires >=8 chars with at least 1 letter and 1 digit;
    # token_urlsafe's alphabet includes both but isn't guaranteed to contain
    # each in every draw, so make it explicit rather than relying on chance.
    password = f"Bk{secrets.token_urlsafe(24)}9"
    register_resp = client.post(
        "/v1/auth/register",
        json={
            "username": username,
            "password": password,
            "token": pseudo_data["token"],
            "pseudonym": pseudo_data["pseudonym"],
            "email": f"{username}@local.berkie-reachy",
        },
    )
    register_resp.raise_for_status()
    access_token = register_resp.json()["tokens"]["access"]["token"]
    return username, password, access_token


def _login(client: httpx.Client, username: str, password: str) -> str:
    resp = client.post("/v1/auth/login", json={"username": username, "password": password})
    resp.raise_for_status()
    return resp.json()["tokens"]["access"]["token"]


def _conversation_still_valid(client: httpx.Client, token: str, conversation_id: str) -> bool:
    try:
        resp = client.get(
            f"/v1/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_seeded(
    base_url: str,
    *,
    existing: Optional[SeedResult],
    persist_credentials: Callable[[str, str], None],
    # This becomes the literal "## Event topic" line in every prompt
    # (eventQuestionHandler.ts: `const topic = options?.topic || this.conversation.name`
    # - there's no separate description field fed to the LLM) - naming it
    # generically meant Berky had no way to know "BKC" means Berkman Klein
    # Center short of a live web search happening to resolve it (it didn't;
    # BKC is also a Mumbai business district, so search wasn't reliable
    # here anyway). Spell it out so it's always in context.
    conversation_name: str = "Berkman Klein Center (BKC) - Berky Reachy Live",
    timeout: float = 15.0,
) -> SeedResult:
    """Ensure a Berky operator account, topic, and conversation exist; return their details.

    ``existing`` is whatever was previously persisted (e.g. read from instance .env) -
    if it still resolves, this is a no-op. ``persist_credentials(username, password)``
    is called immediately after a fresh registration succeeds (before topic/conversation
    creation), so a mid-seed crash doesn't lose the account and cause a duplicate
    registration on retry.
    """
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        if existing is not None:
            try:
                token = _login(client, existing.username, existing.password)
                if _conversation_still_valid(client, token, existing.conversation_id):
                    logger.info("Reusing existing Berky conversation %s", existing.conversation_id)
                    return existing
            except httpx.HTTPError:
                pass  # fall through to re-seed

        if existing is not None:
            try:
                token = _login(client, existing.username, existing.password)
                username, password = existing.username, existing.password
            except httpx.HTTPError:
                username, password, token = _register_operator(client)
                persist_credentials(username, password)
        else:
            username, password, token = _register_operator(client)
            persist_credentials(username, password)

        headers = {"Authorization": f"Bearer {token}"}

        topic_resp = client.post(
            "/v1/topics",
            headers=headers,
            json={
                "name": conversation_name,
                "votingAllowed": False,
                "conversationCreationAllowed": True,
                "private": False,
                "archivable": False,
            },
        )
        if topic_resp.status_code >= 400:
            raise SeedError(f"Failed to create topic: {topic_resp.status_code} {topic_resp.text}")
        topic_id = topic_resp.json()["id"]

        conversation_resp = client.post(
            "/v1/conversations",
            headers=headers,
            json={
                "name": conversation_name,
                "topicId": topic_id,
                # Channels auto-generate a passcode unless explicitly disabled; this
                # conversation is exclusively for Berky's own local backend, not a
                # shared/public channel, so there's nothing to protect against.
                "channels": [{"name": "transcript", "passcode": None}, {"name": "chat", "passcode": None}],
                "agentTypes": [
                    {
                        "name": "voiceAssistant",
                        "properties": {
                            "agentConfig": DEFAULT_AGENT_CONFIG,
                            "llmPlatform": DEFAULT_LLM_PLATFORM,
                            "llmModel": DEFAULT_LLM_MODEL,
                        },
                    }
                ],
            },
        )
        if conversation_resp.status_code >= 400:
            raise SeedError(f"Failed to create conversation: {conversation_resp.status_code} {conversation_resp.text}")
        conversation_id = conversation_resp.json()["id"]

        logger.info("Seeded new Berky conversation %s", conversation_id)
        return SeedResult(conversation_id=conversation_id, username=username, password=password)
