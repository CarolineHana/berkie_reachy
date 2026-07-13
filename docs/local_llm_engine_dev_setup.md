# Local LLM Engine Dev Setup (Berky)

Context: `berkie_reachy`'s `.env` (`BERKIE_LLM_ENGINE_CONVERSATION_ID`,
`BERKIE_LLM_ENGINE_USERNAME`/`PASSWORD`) previously pointed at a conversation ID
and credentials that don't exist in `llm_engine`'s local dev MongoDB
(`mongodb://127.0.0.1:27017/llm_engine`) — likely leftover from a different
environment. This doc records the working local setup so testing doesn't
require finding the real production conversation ID.

## Current state (as of 2026-07-13)

- `llm_engine` repo: `/Users/carolinehana/llm_engine`
- Local Mongo: `mongodb://127.0.0.1:27017/llm_engine`
- Berky agent document (`baseusers` collection, `_id: 6a406e1ee5e0d35f173c446b`,
  `name: "Berky"`) is configured with:
  - `agentType: "voiceAssistant"`
  - `agentConfig.personality: "sarcastic-expert"`
  - `agentConfig.tools: ["web_search"]`
  - matches the contract in [`berky_llm_engine_contract.md`](./berky_llm_engine_contract.md)
- That agent is attached to conversation `"Berky Reachy Live Test"`
  (`_id: 6a406e1ee5e0d35f173c4469`), which has `active: true`, `enableAgents: true`.
- `berkie_reachy/.env` now points `BERKIE_LLM_ENGINE_CONVERSATION_ID` at that
  conversation ID.
- A local admin user `berkyadmin` (matching the username already in `.env`)
  was registered in the local DB via `/v1/auth/register` so the existing
  `BERKIE_LLM_ENGINE_USERNAME`/`BERKIE_LLM_ENGINE_PASSWORD` in `.env` work
  against this local instance without needing to change them.

## Running llm_engine locally

```bash
cd /Users/carolinehana/llm_engine
npm run dev
```

This starts the dev server (nodemon + ts-node) on:
- `PORT=3000` → REST API, matches `BERKIE_LLM_ENGINE_BASE_URL=http://localhost:3000/v1`
- `WEBSOCKET_BASE_PORT=5555` → Socket.IO, matches `BERKY_LLM_ENGINE_SOCKET_URL=http://localhost:5555`

Health check: `curl http://localhost:3000/v1/health` should return `{"status":"OK",...}`.

**The dev server does not persist across shell/session restarts** — if it's
not running, `berky-reachy` will fail to connect. Start it before testing.

## If you need to recreate the local admin user

If the local Mongo gets reset and login starts failing again:

```bash
# 1. Get a registration token
curl -s http://localhost:3000/v1/auth/newPseudonym
# -> {"token": "...", "pseudonym": "..."}

# 2. Register using the username/password already in berkie_reachy/.env
curl -s -X POST http://localhost:3000/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"<BERKIE_LLM_ENGINE_USERNAME>","password":"<BERKIE_LLM_ENGINE_PASSWORD>","token":"<token>","pseudonym":"<pseudonym>","email":"berkyadmin@local.test"}'
```

## Finding the real production conversation ID (unresolved)

This local setup is a workaround for testing, not the fix for production. The
actual robot deployment's `.env` may point at a different, real production
`llm_engine` + MongoDB. To find the real production conversation ID, check
(in order of authority):

1. The `.env` on the machine that actually runs `berky-reachy` at the live event.
2. Nextspace (referenced in `berky_llm_engine_contract.md`) — the conversation
   is normally created there per event.
3. Whatever hosts the production `llm_engine` service (its `MONGODB_URL` secret).
4. Whoever set up the specific Berkman Klein Center event deployment.

Once the real conversation ID is known, the same `agentType`/`personality`/`tools`
fix documented above needs to be applied to the production Berky agent document,
not just the local one.
