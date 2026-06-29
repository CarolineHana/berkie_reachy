---
title: Berkie Reachy
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Berkie Reachy

Forked from the Reachy Mini conversation app.

Berky is an embodied AI agent running on a Reachy humanoid robot at the
Berkman Klein Center's Applied Social Media Lab.

The preferred runtime is `berky-reachy`:

1. Reachy microphone audio is transcribed locally with Whisper.
2. Final transcript chunks are sent to LLM Engine over Socket.IO on the
   `transcript` channel.
3. The backend Berky agent decides when to respond, including the wake phrase
   "hey berkie".
4. Agent responses arriving on the configured response channel are spoken by
   Reachy through local TTS.

## LLM Engine setup

Create or select an LLM Engine conversation in Nextspace with a Berky-like
agent attached. The closest existing backend reference is
`src/agents/eventAssistant/voiceAssistant.ts`: it already listens to the
`transcript` channel, fuzzy-matches wake phrases, and routes responses back to
a chat channel.

Set these environment variables in `.env`:

```bash
BERKIE_LLM_ENGINE_BASE_URL=http://localhost:3000/v1
BERKY_LLM_ENGINE_SOCKET_URL=http://localhost:5555
BERKIE_LLM_ENGINE_CONVERSATION_ID=...
BERKIE_LLM_ENGINE_TOKEN=...

BERKY_TRANSCRIPT_CHANNEL=transcript
BERKY_RESPONSE_CHANNELS=chat
BERKY_WAKE_PHRASE="hey berkie"
```

Instead of `BERKIE_LLM_ENGINE_TOKEN`, you can provide
`BERKIE_LLM_ENGINE_USERNAME` and `BERKIE_LLM_ENGINE_PASSWORD`; the app will log
in to `/v1/auth/login` and use the returned access token.

Optional settings:

- `BERKY_TRANSCRIPT_CHANNEL_PASSCODE`: passcode for a protected transcript channel.
- `BERKY_WHISPER_MODEL`: local faster-whisper model, default `base.en`.
- `BERKY_SPEECH_RMS_THRESHOLD`, `BERKY_SILENCE_SECONDS`, `BERKY_TRANSCRIBE_WINDOW_SECONDS`: audio segmentation tuning.
- `BERKY_TTS_COMMAND`: local TTS command. Use `{text}` as the utterance placeholder.

Run the Reachy-side runtime with:

```bash
pip install -e ".[berky_voice]"
berky-reachy
```

## Virtual Reachy testing

First smoke-test the lightweight Reachy mockup daemon without Whisper or LLM
Engine. This is the best path for local development because it does not require
MuJoCo or graphics acceleration:

```bash
berky-reachy --mockup-sim --robot-smoke-test
```

Then run the Berky client against the mockup daemon:

```bash
berky-reachy --mockup-sim
```

The heavier MuJoCo simulator remains available when installed and supported by
your machine:

```bash
berky-reachy --virtual-reachy --media-backend no_media --robot-smoke-test
```

If the daemon is already running, omit `--mockup-sim`/`--virtual-reachy` and
point at it:

```bash
berky-reachy --host localhost --port 8000
```

The original OpenAI realtime conversation app remains available as
`berkie-reachy`, but it is not the main Berky meeting architecture.

Use the `src/berkie_reachy/profiles/_berkie_reachy_locked_profile` folder to customize your own app from this template:
- Edit instructions `_berkie_reachy_locked_profile/instructions.txt`
- Edit available tools in `_berkie_reachy_locked_profile/tools.txt`
- You can create your own tools in `_berkie_reachy_locked_profile` by subclassing the `Tool` class.

Do not forget to customize:
- this `README.md` file
- the `index.html` file (Hugging Face Spaces landing page)
- the `src/berkie_reachy/static/index.html` (the web app parameters page)

The original README from the conversation app is available in `README_OLD.md`.
