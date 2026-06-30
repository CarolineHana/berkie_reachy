# Berky LLM Engine Contract

Berky's backend agent is the existing `voiceAssistant` agent type in LLM Engine
(`src/agents/eventAssistant/voiceAssistant.ts`). No dedicated agent type is needed.

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

`voiceAssistant` already handles everything Berky needs:

- Listens to finalized transcript messages on the `transcript` channel.
- Fuzzy-matches the wake phrase `hey berkie` / `hey berky` and the configured `agentConfig.botName`.
- Two-turn detection: a bare wake phrase activates the next transcript turn.
- Calls `answerQuestion` from `eventQuestionHandler.ts` for RAG over the live transcript, speaker bios, and background resources.
- Personality injected via `agentConfig.personality` (currently `'sarcastic-expert'`).
- Tool-augmented answers via `agentConfig.tools` (currently `['web_search']`).
- Responds on the `chat` channel with `bodyType: 'json'` and `body.source: 'voice'`.

## Reachy Client Filtering

`llm_engine_socket.py` (`_is_relevant_agent_message`) filters incoming agent messages to only those with `bodyType: 'json'` and `body.source === 'voice'`, so Berky does not speak check-ins, intros, or messages from other agents.

## Agent Configuration (MongoDB `baseusers`)

```json
{
  "agentType": "voiceAssistant",
  "agentConfig": {
    "botName": "Berkie",
    "personality": "sarcastic-expert",
    "tools": ["web_search"]
  }
}
```

## Conversation Configuration (MongoDB `conversations`)

Set `name` and `description` on the conversation document to give the LLM context about the event. Add `presenters`, `moderators`, and `resources` for RAG grounding.
