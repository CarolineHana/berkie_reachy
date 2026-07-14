"""Socket.IO client for sending live Berky transcripts to LLM Engine."""

from __future__ import annotations

import uuid
import json
import time
import base64
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

import httpx

from berkie_reachy.config import config


logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
AgentMessageCallback = Callable[[JsonDict], Awaitable[None]]


def _strip(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _api_url(path: str) -> str:
    return f"{_strip(config.BERKIE_LLM_ENGINE_BASE_URL).rstrip('/')}/{path.lstrip('/')}"


def _message_text(message: JsonDict) -> str:
    body = message.get("body")
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, dict):
        for key in ("text", "message", "answer"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if body is not None:
        return str(body).strip()
    return ""


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        message = data.get("message") or data.get("error")
        if isinstance(message, str) and message.strip():
            return message.strip()
    text = response.text.strip()
    return text[:500] if text else response.reason_phrase


def _jwt_claims(token: str) -> JsonDict | None:
    """Decode JWT claims without verifying; auth still happens server-side."""
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _jwt_subject(token: str) -> str | None:
    """Decode a JWT subject without verifying; auth still happens server-side."""
    data = _jwt_claims(token)
    sub = data.get("sub") if data else None
    return sub.strip() if isinstance(sub, str) and sub.strip() else None


def _jwt_expiry(token: str) -> float | None:
    """Decode a JWT's expiry (epoch seconds) without verifying, or None if absent/unparseable."""
    data = _jwt_claims(token)
    exp = data.get("exp") if data else None
    return float(exp) if isinstance(exp, (int, float)) else None


def _transcript_channel() -> dict[str, str]:
    channel = {"name": config.BERKY_TRANSCRIPT_CHANNEL}
    passcode = _strip(config.BERKY_TRANSCRIPT_CHANNEL_PASSCODE)
    if passcode:
        channel["passcode"] = passcode
    return channel


# Re-authenticate this long before expiry so a slow login attempt (or one retry)
# still finishes before the previous token actually expires.
_TOKEN_REFRESH_MARGIN_SECONDS = 5 * 60


@dataclass(frozen=True)
class AuthSession:
    """Authenticated LLM Engine session details."""

    token: str
    user_id: str
    expires_at: float | None  # epoch seconds; None if unknown (e.g. unparseable token)


class LLMEngineSocketClient:
    """Thin client over LLM Engine's existing Socket.IO message API."""

    def __init__(self, *, on_agent_message: AgentMessageCallback) -> None:
        try:
            import socketio
        except ImportError as exc:
            raise RuntimeError("Install python-socketio to use the Berky LLM Engine socket client.") from exc

        self.on_agent_message = on_agent_message
        self.session: AuthSession | None = None
        self.sio = socketio.AsyncClient(
            reconnection=True,
            logger=False,
            engineio_logger=False,
        )
        self._connected = asyncio.Event()
        self._initial_connect_done = False
        self._refresh_task: asyncio.Task[None] | None = None
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sio.event
        async def connect() -> None:
            logger.info("Connected to LLM Engine Socket.IO at %s", config.BERKY_LLM_ENGINE_SOCKET_URL)
            self._connected.set()
            if self._initial_connect_done and self.session is not None:
                asyncio.create_task(self.join_conversation())

        @self.sio.event
        async def disconnect() -> None:
            logger.warning("Disconnected from LLM Engine Socket.IO")
            self._connected.clear()

        @self.sio.on("message:new")
        async def message_new(message: JsonDict) -> None:
            if not self._is_relevant_agent_message(message):
                return
            await self.on_agent_message(message)

        @self.sio.on("error")
        async def error(data: JsonDict) -> None:
            logger.error("LLM Engine socket error: %s", data)

    async def authenticate(self) -> AuthSession:
        """Authenticate with LLM Engine using token or username/password config."""
        configured_token = _strip(config.BERKIE_LLM_ENGINE_TOKEN)
        if configured_token:
            user_id = await self._fetch_user_id(configured_token)
            expires_at = _jwt_expiry(configured_token)
            if expires_at is not None:
                logger.warning(
                    "Using a static BERKIE_LLM_ENGINE_TOKEN that expires; it cannot be "
                    "auto-refreshed like a username/password login. Use "
                    "BERKIE_LLM_ENGINE_USERNAME/PASSWORD instead for long-running sessions."
                )
            return AuthSession(token=configured_token, user_id=user_id, expires_at=expires_at)

        username = _strip(config.BERKIE_LLM_ENGINE_USERNAME)
        password = _strip(config.BERKIE_LLM_ENGINE_PASSWORD)
        if not username or not password:
            raise RuntimeError(
                "Set BERKIE_LLM_ENGINE_TOKEN or BERKIE_LLM_ENGINE_USERNAME/"
                "BERKIE_LLM_ENGINE_PASSWORD for LLM Engine authentication."
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                _api_url("auth/login"),
                json={"username": username, "password": password},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"LLM Engine login failed: {_error_text(response)}")
            data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError("LLM Engine login returned a non-object response.")

        token = None
        tokens = data.get("tokens")
        if isinstance(tokens, dict):
            access = tokens.get("access")
            if isinstance(access, dict):
                token = access.get("token")
        user = data.get("user")
        user_id = user.get("id") if isinstance(user, dict) else None

        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("LLM Engine login response did not include an access token.")
        if not isinstance(user_id, str) or not user_id.strip():
            user_id = await self._fetch_user_id(token)

        token = token.strip()
        return AuthSession(token=token, user_id=user_id.strip(), expires_at=_jwt_expiry(token))

    async def _fetch_user_id(self, token: str) -> str:
        """Resolve the current user id from a JWT token."""
        subject = _jwt_subject(token)
        if subject:
            return subject

        # Validate the token when we cannot inspect it. LLM Engine currently
        # returns only "pong" here, so this is a sanity check before falling
        # back to a stable room id for error events.
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(_api_url("auth/ping"), headers={"Authorization": f"Bearer {token}"})
            if response.status_code >= 400:
                raise RuntimeError(f"LLM Engine token validation failed: {_error_text(response)}")

        return "berky-reachy"

    async def connect(self) -> None:
        """Authenticate, connect, and join configured LLM Engine rooms."""
        if not _strip(config.BERKIE_LLM_ENGINE_CONVERSATION_ID):
            raise RuntimeError("BERKIE_LLM_ENGINE_CONVERSATION_ID is required.")

        self.session = await self.authenticate()
        await self.sio.connect(config.BERKY_LLM_ENGINE_SOCKET_URL, transports=["websocket"])
        await asyncio.wait_for(self._connected.wait(), timeout=10.0)
        await self.join_conversation()
        self._initial_connect_done = True

        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def _refresh_loop(self) -> None:
        """Re-authenticate before the access token expires.

        LLM Engine's socket auth middleware verifies the JWT on every event
        (not just the initial join), so a session left running past the
        token's lifetime (JWT_ACCESS_EXPIRATION_MINUTES, 30 min by default)
        has every subsequent transcript silently rejected server-side - Berky
        just stops responding, with no error surfaced to this client. Long
        meetings routinely exceed that, so keep the token fresh in the
        background for as long as the socket client is alive.

        A statically configured BERKIE_LLM_ENGINE_TOKEN can't be renewed this
        way (there's no username/password to re-login with, so re-running
        authenticate() would just hand back the same expiring token) -
        authenticate() already warns about that case, so this loop is a
        no-op when that's the configured auth mode.
        """
        if _strip(config.BERKIE_LLM_ENGINE_TOKEN):
            return

        try:
            while True:
                session = self.session
                if session is None or session.expires_at is None:
                    # Nothing to schedule against yet; check back shortly.
                    await asyncio.sleep(_TOKEN_REFRESH_MARGIN_SECONDS)
                    continue

                sleep_for = max(0.0, session.expires_at - time.time() - _TOKEN_REFRESH_MARGIN_SECONDS)
                await asyncio.sleep(sleep_for)

                try:
                    self.session = await self.authenticate()
                    logger.info("Refreshed LLM Engine auth token before expiry")
                except Exception:
                    logger.exception("Failed to refresh LLM Engine auth token; will retry shortly")
                    await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            pass

    async def join_conversation(self) -> None:
        """Join transcript and response rooms for the configured conversation."""
        if self.session is None:
            raise RuntimeError("Cannot join conversation before authentication.")

        channels = [_transcript_channel(), *[{"name": name} for name in config.BERKY_RESPONSE_CHANNELS]]
        payload = {
            "token": self.session.token,
            "userId": self.session.user_id,
            "conversationId": config.BERKIE_LLM_ENGINE_CONVERSATION_ID,
            "channels": channels,
        }

        ack_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def _ack(data: Any = None) -> None:
            if not ack_future.done():
                ack_future.set_result(data)

        await self.sio.emit("conversation:join", payload, callback=_ack)
        try:
            await asyncio.wait_for(ack_future, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for conversation:join acknowledgement")

    async def send_transcript(self, text: str, *, final: bool = True, speaker: str | None = None) -> None:
        """Send one transcript message to LLM Engine."""
        if self.session is None:
            raise RuntimeError("Cannot send transcript before connect().")

        if not self.sio.connected:
            logger.warning("Socket not connected, skipping transcript (will resume on reconnect)")
            return

        clean_text = text.strip()
        if not clean_text:
            return

        source: JsonDict = {
            "type": "berky_reachy_transcript",
            "final": final,
            "requestId": str(uuid.uuid4()),
        }
        if speaker:
            source["speaker"] = speaker

        payload = {
            "token": self.session.token,
            "userId": self.session.user_id,
            "request": source["requestId"],
            "message": {
                "body": clean_text,
                "bodyType": "text",
                "conversation": config.BERKIE_LLM_ENGINE_CONVERSATION_ID,
                "channels": [_transcript_channel()],
                "source": source,
            },
        }
        await self.sio.emit("message:create", payload)

    async def disconnect(self) -> None:
        """Close the Socket.IO connection."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self.sio.connected:
            await self.sio.disconnect()

    def _is_relevant_agent_message(self, message: JsonDict) -> bool:
        if not message.get("fromAgent"):
            return False

        # voiceAssistant posts JSON body with source='voice'.
        # Filter to only those so Reachy doesn't speak check-ins, intros, etc.
        body = message.get("body")
        if message.get("bodyType") == "json" and isinstance(body, dict):
            if body.get("source") != "voice":
                return False

        if not _message_text(message):
            return False
        channels = message.get("channels")
        if not isinstance(channels, list):
            return True
        return any(channel in config.BERKY_RESPONSE_CHANNELS for channel in channels)


__all__ = ["LLMEngineSocketClient", "AuthSession", "_message_text"]
