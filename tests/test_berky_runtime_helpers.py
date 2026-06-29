from __future__ import annotations

import json
import base64

import numpy as np

from berkie_reachy.tts import CommandTTS
from berkie_reachy.local_whisper import _mono_float32
from berkie_reachy.llm_engine_socket import _jwt_subject, _message_text


def _jwt_with_subject(subject: str) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": subject}).encode()).decode().rstrip("=")
    return f"{header}.{payload}."


def test_jwt_subject_decodes_sub_claim() -> None:
    assert _jwt_subject(_jwt_with_subject("user-123")) == "user-123"


def test_message_text_extracts_string_and_json_bodies() -> None:
    assert _message_text({"body": "hello"}) == "hello"
    assert _message_text({"body": {"text": "from json"}}) == "from json"
    assert _message_text({"body": {"answer": "fallback answer"}}) == "fallback answer"


def test_command_tts_replaces_text_placeholder() -> None:
    tts = CommandTTS("say --voice Alex {text}")
    assert tts._argv("hello Berky") == ["say", "--voice", "Alex", "hello Berky"]


def test_mono_float32_downmixes_and_scales_int16() -> None:
    stereo = np.array([[0, 100], [32767, -32767]], dtype=np.int16)
    mono = _mono_float32(stereo)
    assert mono.dtype == np.float32
    assert mono.shape == (2,)
    assert mono[0] == 0.0
    assert 0.99 < float(mono[1]) <= 1.0
