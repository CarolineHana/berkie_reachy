# Local LLM Engine Dev Setup (Berky)

Context: `berkie_reachy`'s `.env` (`BERKIE_LLM_ENGINE_CONVERSATION_ID`,
`BERKIE_LLM_ENGINE_USERNAME`/`PASSWORD`) previously pointed at a conversation ID
and credentials that don't exist in `llm_engine`'s local dev MongoDB
(`mongodb://127.0.0.1:27017/llm_engine`) — likely leftover from a different
environment. This doc records the working local setup, the bugs found and
fixed along the way, and what's still unresolved for production.

## Current state (as of 2026-07-13)

- `llm_engine` repo: `/Users/carolinehana/llm_engine`
- Local Mongo: `mongodb://127.0.0.1:27017/llm_engine`
- Berky agent document (`baseusers` collection, `_id: 6a406e1ee5e0d35f173c446b`,
  `name: "Berky"`) is configured with:
  - `agentType: "voiceAssistant"`
  - `agentConfig.personality: "sarcastic-expert"`
  - `agentConfig.tools: ["web_search"]`
  - `llmPlatform: "openai"`, `llmModel: "gpt-5.2-2025-12-11"` (see Bedrock
    blocker below for why this isn't `bedrock`)
  - matches the contract in [`berky_llm_engine_contract.md`](./berky_llm_engine_contract.md)
    except for the `llmPlatform` override
- That agent is attached to conversation `"Berky Reachy Live Test"`
  (`_id: 6a406e1ee5e0d35f173c4469`), which has `active: true`, `enableAgents: true`.
- `berkie_reachy/.env` points `BERKIE_LLM_ENGINE_CONVERSATION_ID` at that
  conversation ID.
- A local admin user `berkyadmin` (matching the username already in `.env`)
  was registered in the local DB via `/v1/auth/register` so the existing
  `BERKIE_LLM_ENGINE_USERNAME`/`BERKIE_LLM_ENGINE_PASSWORD` in `.env` work
  against this local instance without needing to change them.
- **Verified working end-to-end**: posting a wake-phrase message to the
  conversation produces a genuinely sarcastic response on the `chat` channel
  with `bodyType: json`, `body.source: "voice"` — the shape `berkie_reachy`'s
  `_is_relevant_agent_message` filter expects.

## Bugs found and fixed on the Berky agent document

These were all discovered by tracing real failures when testing locally, not
by inspection alone — worth knowing about in case they recur on other agents:

1. **`agentType` was `"berkyAgent"`**, which doesn't exist anywhere in
   `llm_engine`'s code (checked `src/agents/index.ts`'s `agentTypes` registry
   and full git history — never a real agent type). Fixed to `"voiceAssistant"`.
2. **No `agentConfig.personality`** was set, so no personality was applied.
   Fixed to `"sarcastic-expert"` (the only personality currently defined, in
   `src/agents/helpers/agentPersonality.ts`).
3. **No `agentConfig.tools`**. Fixed to `["web_search"]`.
4. **Stale `llmTemplates.user` override**: the document had a persisted
   custom template (`'## Current live meeting context\n{recentTranscript}\n...'`)
   from some earlier/different setup. Current `voiceAssistant` code
   (`eventQuestionHandler.ts`'s `getResponse`) expects `{topic}`/`{context}`/
   `{question}` placeholders, not `{recentTranscript}` — this stale value
   crashed every response with `Missing value for input variable`. Mongoose's
   `agentSchema.pre('validate')` hook (`agent.model/index.ts:466-467`) only
   assigns the code's default template when `llmTemplates` is `undefined`,
   and only runs on `.save()`/`.validate()` — it does **not** re-populate on a
   plain `$unset` via the raw MongoDB driver, so unsetting the field made
   things worse (`Cannot read properties of undefined (reading 'user')`).
   Fixed by setting `llmTemplates.user` directly to the current default
   string (from `buildLLMTemplates()` in `eventQuestionHandler.ts`):
   ```
   ## Event topic:
   {topic}

   ## Context:
   {context}

   ## User question:
   {question}
   ```

## BLOCKER: local Bedrock endpoint is dead (as of 2026-07-13)

Even with the fixes above, Berky produced no response when its `llmPlatform`
was `bedrock`. The `llm_engine` dev server log showed every LLM call failing:

```
Error in fetchFn: Bedrock proxy error: 410 Gone
{"code":410,"message":"The v1 Bedrock API (ais-bedrock-llm/v1) has been
retired and is no longer available. Please migrate to the v2 Bedrock API
(ais-bedrock-llm/v2)... email apihelp@harvard.edu."}
```

`llm_engine/.env`'s `BEDROCK_BASE_URL` is:
```
https://go.stage.apis.huit.harvard.edu/ais-bedrock-llm/v1
```
This is Harvard AIS's **staging** gateway; production may be on a different,
still-working host. A blind `/v1` → `/v2` path swap on the same host returned
`404 Not Found` (an HTML error page, not a JSON API error) — confirming the
v2 API has a different request shape (it "proxies the native AWS Bedrock
runtime and management APIs" per the error message), not just a version bump
in the path. **Do not guess at the v2 URL/request shape** — get the real
endpoint and, if needed, example request format from Harvard AIS
(apihelp@harvard.edu) before touching `src/agents/helpers/claudeHandler.ts`.

**Current workaround**: Berky's agent document has `llmPlatform: "openai"`,
`llmModel: "gpt-5.2-2025-12-11"` instead of the `bedrock` default, using the
already-configured `DEFAULT_OPENAI_BASE_URL`/`DEFAULT_OPENAI_API_KEY` in
`.env`. This is scoped to just this one agent document, not a global config
change. Once real v2 Bedrock credentials/docs are obtained, switch back via:

```js
db.baseusers.updateOne(
  { _id: ObjectId("6a406e1ee5e0d35f173c446b") },
  { $set: { llmPlatform: "bedrock", llmModel: "us.anthropic.claude-opus-4-6-v1" } }
)
```

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

**Also needs local ChromaDB** (used for RAG over the event transcript, via
`CHROMA_DB_URL=http://localhost:8002` in `llm_engine/.env`). If not running,
responses fail with `ChromaConnectionError`. Start it with:

```bash
chroma run --port 8002
```

If RAG finds no docs (`Could not find relevant RAG docs from ...`), that's
expected/non-fatal for a fresh local Chroma with no transcript indexed yet —
the agent still responds using general knowledge + tools.

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

To post a test message directly via REST (bypassing `berky-reachy`/audio):

```bash
# channels must be an array of objects, not strings: [{"name": "transcript"}]
curl -s -X POST http://localhost:3000/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access token from login/register response>" \
  -d '{"body":"Hey berkie, <question>","bodyType":"text","conversation":"6a406e1ee5e0d35f173c4469","channels":[{"name":"transcript"}]}'
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

Once the real conversation ID is known, the same `agentType`/`personality`/
`tools`/`llmTemplates` fixes documented above need to be checked/applied to
the production Berky agent document too — this local fix does not touch
production. Also confirm production's Bedrock endpoint isn't hitting the same
v1-retired issue (production may already be on a working v2 or non-staging
host, but worth checking, not assuming).
