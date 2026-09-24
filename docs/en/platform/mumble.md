# Connect Mumble

The Mumble adapter connects to a Mumble server (1.5 or later) as a bot, sending and receiving text in channels and private messages. With realtime voice on, it also talks in channels by voice (full duplex), powered by Codex realtime voice.

## Create the Mumble platform adapter

Open the `Bots` page, click `+ Create Bot`, choose `Mumble`, turn on `Enable` and fill in:

- `Mumble Server` and `Mumble Port` (default `64738`)
- `Bot Username`: the name the bot shows in Mumble
- `Server Password`: if the server has one, or the password of the bot's registered account
- `Initial Channel`: channel name or path from the root (e.g. `Games/Lobby`) to join; empty stays in the default channel
- `Channel Text Wake Prefix`: `!` by default, see below

On first connect a client certificate is generated under `data/mumble/<platform id>/`, so the server recognises the bot and an admin can register it and grant permissions in the Mumble client. You can also point `Client Certificate` at your own.

## Conversations and waking

- **The whole server is one group conversation** (session id `server`), whichever channel the bot is in, so an admin can move the bot at any time; replies go to the channel the message came from.
- **Private messages** are one conversation per user. The user id is the client certificate hash (official clients always have one); use it when configuring admins. Users with neither a certificate nor a registration only keep their id for the current connection.
- Mumble has no @ mentions. **Channel text starting with the wake prefix** (e.g. `!what's the weather`) wakes the bot, like @-ing it; private messages always do. The global wake prefix (e.g. `/` commands) still works.

## Realtime voice

Requires the `Codex agent runner` signed in with a **ChatGPT plan that includes voice**. Voice uses Codex realtime over WebRTC; no API key is needed.

- It joins in when someone speaks in the bot's channel. In channels it **only answers when called by name** (enforced by the prompt, so it may occasionally chime in). **Whispers** to the bot are always answered, and answered by whisper.
- Tasks asked by voice are **handled by the agent of the paired conversation**: the server group for channel voice, the user's private chat for whispers. Each runs as a turn of that conversation, sharing its context, persona, tools, memories and approvals with the text chat; text messages and voice requests queue behind each other, and while the conversation is busy the bot says so first and gives the result when it is done. Answers are only spoken, never posted to the chat. Identity: channel voice cannot tell speakers apart, so it always runs as a fixed voice user locked to the member role; whispers run as the speaker, with their own permissions. (With `unique_session` on, group chats are split per user, while channel voice still pairs with the single server-wide group conversation.)
- **Standby**: after `Voice Standby Timeout` (default 300 s) without recognised speech, realtime voice disconnects; new speech resumes it automatically, including the words that woke it.
- **Mute / unmute**: send `wake prefix + mute` (or `闭麦`, or with the global wake prefix) in a channel, or just `mute` in a private message, and the bot stops listening and speaking and shows as muted; `unmute` (`开麦`) restores it. Muted for over a minute, it drops its voice sessions.
- Voice options: `Voice Wake Name` and `Voice Wake Aliases`, `Voice` (juniper, maple, spruce, ember, vale, breeze, arbor, sol, cove; default cove), `Realtime Model`, `Extra Voice Prompt`. When the paired conversation's active persona has a voice persona, the voice model uses it instead of the extra voice prompt.

## Local voice backend (MiniCPM-o)

With `Voice Backend` set to **Local MiniCPM-o**, the full-duplex conversation runs on your own MiniCPM-o 4.5 server instead; no ChatGPT plan is needed. Lookups and actions still go to the paired conversation's agent (Codex) described above.

- **Server**: the `astrbot-omni` branch of llama.cpp-omni (voice clone, forced speech and a tool router on top of upstream), with `MiniCPM-o-4_5-Q4_K_M.gguf` and the audio / tts / token2wav GGUFs. Recommended flags: `-ngl 99 -c 8192 -fa on -ctk q8_0 -ctv q8_0 --temp 0.7 --top-k 100 --top-p 0.8 --repeat-penalty 1.05`, with `OMNI_TTS_N_CTX=4096` and `OMNI_ROUTER_CTX=2048` in the environment; for voice cloning also set `OMNI_VOICE_BUNDLE_CMD` to run `tools/omni/voice/make_voice_bundle.py` (needs `campplus.onnx` and `speech_tokenizer_v2_25hz.onnx`). Measured at about 9.7 GB of VRAM; keep the context at 8192 (what the model was trained on; answers get worse beyond it). The server slides the window past that, keeping the last few minutes. A 12 GB card runs it.
- Set `MiniCPM-o Server URL` to its WebSocket URL (e.g. `ws://127.0.0.1:19060/backend`). The server serves one voice conversation at a time: channel voice and whispers are first come, first served; others are not answered.
- **Tool calls**: in full-duplex mode the model never calls tools. The server decides every utterance with the same model in text mode (Qwen3 tool format): `reply` lets the model answer; `silence` means it was not said to the bot, which stays quiet; `backend_task` hands the task to the paired conversation's agent, and the model reads its answer aloud verbatim in its own voice. Utterances are found and transcribed locally in AstrBot (Silero VAD + SenseVoice on CPU; about 240 MB of models are downloaded on first use).
- `Channel Silence Bias` sets how readily channel speech is taken as not meant for the bot (default 4; higher answers less); whispers are not affected. `Task Acknowledgement` is what the bot says when it hands a task over.
- **Voice clone**: set `Reference Audio` to a local audio file; its first 10 seconds are used, and 5 to 10 seconds of clear speech work best.

## Proxy

Set `Codex Proxy` in the Codex agent runner settings (e.g. `socks5://127.0.0.1:7890` or `http://127.0.0.1:7890`) and all of Codex's traffic uses it: model requests, sign-in, and realtime voice signalling and control. The rest of AstrBot is unaffected.

Realtime voice media normally uses UDP, which proxies cannot carry. With a proxy set, the adapter switches to the peer's ICE-TCP channel on port 443 and sends voice over one TCP connection opened through the proxy; no UDP port has to be open.

## Run a Mumble server

The official image works as is:

```bash
docker run -d --name mumble-server -p 64738:64738/tcp -p 64738:64738/udp mumblevoip/mumble-server
```

The first start logs the `SuperUser` password. The bot only uses TCP (voice is tunnelled over it), so no extra ports are needed.
