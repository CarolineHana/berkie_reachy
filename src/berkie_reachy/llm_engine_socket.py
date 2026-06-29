"""Socket.IO client for sending live Berky transcripts to LLM Engine."""

from __future__ import annotations

import uuid
import json
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


def _jwt_subject(token: str) -> str | None:
    """Decode a JWT subject without verifying; auth still happens server-side."""
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
    sub = data.get("sub") if isinstance(data, dict) else None
    return sub.strip() if isinstance(sub, str) and sub.strip() else None


def _transcript_channel() -> dict[str, str]:
    channel = {"name": config.BERKY_TRANSCRIPT_CHANNEL}
    passcode = _strip(config.BERKY_TRANSCRIPT_CHANNEL_PASSCODE)
    if passcode:
        channel["passcode"] = passcode
    return channel


@dataclass(frozen=True)
class AuthSession:
    """Authenticated LLM Engine session details."""

    token: str
    user_id: str


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
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sio.event
        async def connect() -> None:
            logger.info("Connected to LLM Engine Socket.IO at %s", config.BERKY_LLM_ENGINE_SOCKET_URL)
            self._connected.set()

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
            return AuthSession(token=configured_token, user_id=user_id)

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

        return AuthSession(token=token.strip(), user_id=user_id.strip())

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
        if self.sio.connected:
            await self.sio.disconnect()

    def _is_relevant_agent_message(self, message: JsonDict) -> bool:
        if not message.get("fromAgent"):
            return False
        if not _message_text(message):
            return False
        channels = message.get("channels")
        if not isinstance(channels, list):
            return True
        return any(channel in config.BERKY_RESPONSE_CHANNELS for channel in channels)


__all__ = ["LLMEngineSocketClient", "AuthSession", "_message_text"]
