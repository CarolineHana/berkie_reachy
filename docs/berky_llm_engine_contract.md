# Berky LLM Engine Contract

Berky's backend agent should live in LLM Engine, close to
`src/agents/eventAssistant/voiceAssistant.ts`.

## Channels

- Incoming transcript channel: `transcript`
- Outgoing response channel: `chat` by default, configurable through the Reachy client as `BERKY_RESPONSE_CHANNELS`

The Reachy runtime emits Socket.IO `message:create` events with:

```json
{
  "message": {
    "bodyType": "text",
    "body": "Hey Berkie, what did the speaker mean by protocol governance?",
    "channels": [{ "name": "transcript" }],
    "source": {
      "type": "berky_reachy_transcript",
      "final": true,
      "requestId": "..."
    }
  }
}
```

## Agent Behavior

The `berkyAgent` should:

- Listen to finalized transcript messages on the `transcript` channel.
- Fuzzy-match the wake phrase `hey berkie` / `hey berky` and the configured bot name.
- Treat a bare wake phrase as an activation for the next transcript turn, matching the pattern in `voiceAssistant.ts`.
- Use the recent live transcript window as meeting context, including speaker/topic drift when available.
- Use BKC research archive RAG for background knowledge, but do not ignore the live meeting context.
- Respond only when addressed or when the agent has a high-confidence, contextually useful contribution.
- Return a short spoken response on the configured response channel, usually `chat`.

## Suggested Prompt Shape

System prompt:

```text
You are Berky, an embodied AI agent at the Berkman Klein Center's Applied Social Media Lab.
You are physically present through a Reachy humanoid robot in a live meeting.
Use the live transcript as immediate context and BKC research archive retrieval as background knowledge.
When answering, be concise enough to speak aloud. Ground claims in retrieved or live context.
Do not interrupt unless addressed by wake phrase or unless your contribution is clearly useful.
```

User prompt:

```text
Wake-directed question:
{question}

Recent live meeting context:
{recentTranscript}

Relevant BKC archive context:
{ragContext}
```
