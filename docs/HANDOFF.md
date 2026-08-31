# Handoff: Urdu ↔ English live translator

Written 31 August 2026, at commit `efa889b` ("10% working it now"). Working tree
clean, full test suite passing.

This is a working note for whoever picks the project up next. It is written to
save you the day that was spent finding these things out. Read the section
called **Traps** before you change anything — most of what looked like software
faults in this project turned out not to be.

---

## 1. What the application does

A Windows desktop app that sits beside Microsoft Teams and translates a call in
both directions, in real time.

- **Outgoing** — you speak Urdu, the client hears English.
- **Incoming** — the client speaks English, you hear Urdu.

Audio is moved between the app and Teams using virtual audio cables, because
Teams cannot be asked to hand its audio to another program directly.

## 2. Current state, honestly

| Direction | State |
|---|---|
| Yours: Urdu → English, heard by the client | **Working.** Verified on a live call. |
| Client's: English → Urdu, heard by you | **Never tested.** Not known broken — there has never been a second person talking on a call to test it with. |

Nothing in the project costs money to run right now. The whole outgoing path is
free: Deepgram's existing credit, Gemini's free tier, and Edge TTS which needs
no key at all.

Verified working, on a real call:

```
[you]
Original:    میرا نام حشمت علی ہے
Translation: My name is Hashmat Ali.

[you]
Original:    میں میکروساف پاکستان میں کام کرتا ہوں
Translation: I work in Microsoft Pakistan.
```

Note the second one: the recogniser misheard "Microsoft" as `میکروساف`, and the
translator recovered it from context. The rolling-context prompt in
`llm_translator.py` is doing real work — don't strip it out for being verbose.

## 3. How the audio is wired

Two separate cables, one per direction. This is not optional: a single cable
used both ways feeds the app's own output back into its input, which is
recognised as the client speaking, translated back, and played louder on every
pass.

```
YOU SPEAK URDU
  microphone (Microphone Array, Intel)
    → app: Deepgram nova-3 (language=ur)
    → app: Gemini translates to English
    → app: Edge TTS speaks the English
    → CABLE Input  (VB-Audio Virtual Cable)      ← the app writes here
    → CABLE Output (VB-Audio Virtual Cable)      ← Teams' MICROPHONE
    → client hears English

CLIENT SPEAKS ENGLISH
  Teams' SPEAKER = Voicemeeter AUX Input
    → app captures the WASAPI loopback of that endpoint
    → app: Deepgram nova-3 (language=en)
    → app: Gemini translates to Urdu
    → app: Edge TTS speaks the Urdu
    → Speaker (Realtek(R) Audio)                 ← you hear it
```

### Settings that must be right

**Teams → Settings → Devices**

| Setting | Value |
|---|---|
| Microphone | `CABLE Output (VB-Audio Virtual Cable)` |
| Speaker | `Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)` |
| Noise suppression | **Off** — not Auto |
| Spatial audio | Off |

Noise suppression matters more than it sounds. Teams classifies synthesised
speech as noise and removes it, so the client hears nothing while everything in
the app looks perfect.

**Voicemeeter Banana**

- Must be running.
- `A1` (hardware out, top right) should be **empty**, or set through **MME** if
  you want to hear the client's raw voice. Do **not** set it via **WDM** — WDM
  takes the Realtek speaker exclusively and the app then cannot open it for
  Urdu playback, failing with `[Errno -9996] Invalid device`.
- `B1` is lit on the strips, which is why `Voicemeeter Out B1` carries audio.

**The app** sets Windows' default playback and recording devices to the cables
while it runs, and restores them on Stop, on Ctrl+C, and via an `atexit` hook
(`app/audio/routing.py`). You do not need to touch Windows Sound. The only case
it cannot cover is the process being killed outright — then set the default
output back by hand.

## 4. Providers and keys

`.env` in the project root. Read through `get_secret()` in
`app/models/config.py`, which checks the Windows credential store first, then
the environment, then `.env`.

| Key | Used for | Required? |
|---|---|---|
| `DEEPGRAM_API_KEY` | recognition, both directions | **yes** |
| `GEMINI_API_KEY` | translation | one translator key is required |
| `GROQ_API_KEY` | translation (not set up yet) | optional, recommended |
| `OPENAI_API_KEY` | translation, and the realtime model | optional |
| `GEMINI_MODEL` | override the Gemini model | optional |
| `USE_REALTIME` | `1` to use OpenAI's realtime audio model | optional |

### The translator is a chain, not one provider

`build_translator()` in `app/providers/factory.py` builds every translator it
has a key for and wraps them in `ChainTranslator`
(`app/providers/chain_translator.py`), cheapest first: **Groq → Gemini →
OpenAI**. When one returns a quota error the chain moves to the next and says
so. A *real* error — a bad key, a dead model — is raised rather than hidden
behind a second attempt at another vendor.

All of them are the same class. Groq, Gemini and Cloudflare all serve an
OpenAI-compatible endpoint, so adding one is a `base_url` and a model name, not
a new provider:

| Provider | base_url | free allowance |
|---|---|---|
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | per-model daily cap |
| Groq | `https://api.groq.com/openai/v1` | thousands of requests/day |
| Cloudflare Workers AI | `https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1` | 10,000 neurons/day, resets 00:00 UTC |

### The two paths for the outgoing direction

`build_s2st()` in the same file chooses between them:

- **`CascadeSpeechTranslator`** (`app/providers/cascade_s2st.py`) — the default.
  Deepgram → translator → Edge TTS. Billed per sentence of *text*, which is
  pennies. Slower, flatter, and free.
- **`OpenAiSpeechTranslator`** (`app/providers/openai_s2st.py`) — one model,
  audio in and audio out, keeps prosody, sounds much better. Billed per
  **minute of audio**, no free tier. **Opt-in only**, via `USE_REALTIME=1`.

That default is deliberate and it is about money. Leaving the realtime model
automatic means merely having a key turns every minute of testing into spend,
and when the balance empties the outgoing direction dies mid-call looking
exactly like a software fault. That happened, and it cost a day.

Both satisfy the same interface, so nothing above `build_s2st` knows which is in
use.

## 5. Traps

Every one of these was found the hard way. They are the reason this file exists.

**A provider running out of money looks exactly like broken code.** The whole
outgoing direction stopped working mid-session and hours went into audio
routing before the real message was read:
`insufficient_quota.credit_balance_exhausted`. When something that worked stops
working, **check balances and quotas first**.

**Free tiers are smaller than you think.** Google's free tier for
`gemini-2.5-flash` is **twenty requests per day**. One sentence is one request,
so a two-minute conversation exhausts it. Quotas are per *model*, and reset at
midnight Pacific. Two model names are already ruled out and recorded in
`factory.py`: `gemini-2.5-flash` (20/day) and `gemini-2.5-flash-lite` (404, "no
longer available to new users"). Current default: `gemini-3.5-flash-lite`.

**Deepgram hangs up after 10 seconds of silence.** `NET-0001`, "did not receive
audio data or a text message within the timeout window". Ordinary pauses in a
conversation are longer than that. Fixed with a `{"type": "KeepAlive"}` **text**
frame every 4 seconds — `keepalive()` in `deepgram_asr.py`, driven by
`_keep_alive()` in `cascade_s2st.py`. It must be a text frame; a binary one is
documented as mishandled. This was the real cause of "sometimes it hears me,
sometimes it doesn't".

**Windows renames audio devices.** `Speakers (2- VB-Audio Virtual Cable)` became
`Speakers (VB-Audio Virtual Cable)` — the `2-` prefix vanished on its own. The
app pins devices **by name**, so everything silently stopped resolving. If
devices look wrong, run `python run.py --call-check`, which re-detects and saves
them.

**The sink drops audio, silently and by design.** `RingBuffer.write()` never
blocks; on overflow it discards the *oldest* samples. The sink's ring holds
500 ms and the voice delivers a 3.5-second sentence in a fraction of that.
Measured on the real path: 336,384 samples written, **22,080 dropped**. The
client heard a fragment while the transcript showed a perfect translation.
Fixed by `_write_paced()` in `outgoing.py`, which writes only what there is
room for, with a one-second bound so a dead device can never mean permanent
silence. **Do not "simplify" this back to a plain `sink.write()`.**

**`except Exception: pass` cost more time than any other single line.** Both
`deepgram_asr.py` and `openai_s2st.py` swallowed provider errors, so every
failure — bad key, no credit, refused parameter, dropped socket — arrived as the
same silence. Both now keep the failure and re-raise it at the consumer. If you
find another one of these, treat it as a bug.

**"Speech is reaching the recogniser but nothing is coming back. Check the
network and DEEPGRAM_API_KEY."** That message was a guess, and it was wrong
every single time it appeared. The key and network were verified repeatedly. An
idle virtual cable measures about −70.6 dB and the speech gate trips around
−69 dB, so a few frames of *nothing* were being counted as speech. Rewritten in
`engine.py::_watch_for_silence`.

**"Audio is arriving but there is no speech in it"** was also misread for a
whole day. It usually means **nobody on the other end has spoken**. Testing
alone triggers it every run. It is now an info line after 60 seconds rather
than a warning after 12.

**The transcript channel means "the translation".** Everything downstream shows
`transcripts()` output as what the client hears, labelled `[you]`. Pushing
interim source-language text into it filled the window with Urdu fragments that
looked like translation into the wrong language. Only finished, translated text
goes there. `TranscriptDelta.source_text` carries the original alongside it.

**The microphone is weak.** The Intel Smart Sound array measured about **−49 dB**
peak while speaking; a headset sits around −20 to −30. Quiet audio does not fail
loudly — the recogniser guesses. `mere naam Hashmat Ali hy` came back as "my
name is Raj Patel", and near-silence produced the single Urdu word `کے` over and
over. Raise the Windows input level to 100, add microphone boost, turn off
"audio enhancements", and prefer a headset.

## 6. Diagnostic tools

In `tools/`. All of them are read-only with respect to the app — they open their
own streams and change nothing. **Use these before changing code.** Between
them they proved the cables, the key, the network and the playback path, and
they would have found every fault in this project in about twenty minutes.

| Tool | What it proves |
|---|---|
| `python run.py --call-check` | re-detects devices, saves them, prints exactly what Teams must be set to |
| `tools/wire_check.py` | plays a 997 Hz tone into each cable and measures every endpoint that could carry it. Reports level **and purity**, so noise cannot be mistaken for a working wire. Both cables measured −13.6 dB at 0.91 purity. |
| `tools/hear_output.py` | run while the app is speaking: is the English actually on Teams' microphone? |
| `tools/say_english.py` | pushes one real sentence through the exact production path — same TTS, voice, resampling, channel mapping and sink — and measures the far end. This is what found the dropped-samples bug. |
| `tools/check_deepgram.py` | streams live audio to Deepgram and prints every raw reply. Add `--mic` for your own microphone. |
| `tools/check_translator.py` | verifies the translator with **exactly one request**. Use this before starting the app when quota is tight. |
| `tools/find_source.py` | sweeps every endpoint for audio. Superseded by `wire_check`; note it samples each device for only 2 seconds in turn, which is misleading on a live call. |

## 7. What to do next

Roughly in order of value.

**1. Test the incoming direction.** It has never been exercised. Join the
meeting from a phone, talk into the phone, and see whether English appears under
`[client]` with Urdu beneath it. Everything upstream of it is measured and
proven; this is a test that was never run, not a known bug.

**2. Get a free Groq key** (`console.groq.com`) and put `GROQ_API_KEY` in
`.env`. The chain will prefer it and fall back to Gemini automatically, so a
daily cap stops being able to interrupt a call. No code needed.

**3. Improve the outgoing voice.** This is the biggest quality gap and the
owner has explicitly flagged it. Known causes, in order of effect:

- **No jitter buffer.** The model's chunks do not arrive at a steady rate, so
  the device runs dry between them — a stutter per word. A 300 ms buffer before
  the first sample fixes it and costs that latency once per utterance. This was
  written once and lost in a revert; it needs rebuilding.
- **A beat of delay per sentence**, because it waits for Deepgram to declare an
  utterance final before translating. Starting on stable partial hypotheses
  would cut most of it.
- **Flat prosody.** Edge TTS reads the English from scratch, so emphasis and
  emotion do not survive. A better voice (Azure or ElevenLabs, both already
  implemented in `tts.py`) is the cheap improvement; the realtime model is the
  real answer when it can be paid for.
- **Stock voice.** `en-US-AndrewNeural`, set by `english_voice()` in
  `cascade_s2st.py`, overridable with `ENGLISH_VOICE`. The voice-cloning work in
  `voice_clone.py` only applies to the incoming direction.

**4. Add Cloudflare Workers AI to the translator chain.** 10,000 neurons/day
free, OpenAI-compatible, so it is one block in `build_translator` plus
`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in `.env`. Use only
**Cloudflare-hosted** models — the OpenAI and AssemblyAI entries in their
catalogue are third-party, routed to the vendor, and billed by the vendor.

**5. Set Teams' devices from the app**, so the user never opens Teams settings.
Windows has a per-app device assignment — the `Output device` / `Input device`
dropdowns under an app in the Volume mixer — settable through
`IAudioPolicyConfigFactory::SetPersistedDefaultAudioEndpoint`, the same
undocumented COM family `app/audio/routing.py` already uses for the default
device. One caveat: it only takes effect while Teams' own selection is
"Default"; if Teams has a device pinned, Teams wins, and the app should detect
that and say so once.

**6. Verify devices at startup instead of assuming them.** Fold
`app/audio/probe.py` into the Start path: play the tone, confirm it comes back,
and name the exact wire that failed. Every hour lost on this project came from
the software believing a device choice rather than measuring it.

**7. Clean up.** The `[DEBUG OUTGOING]` prints in `outgoing.py` should go behind
an environment flag; they drown the console.

## 8. How to work on this

Learned at some cost:

- **Measure before changing code.** Use the tools in `tools/`. Four of the
  faults in this project were outside the codebase entirely: an empty balance, a
  renamed device, a Teams setting, and a daily quota.
- **Never touch the working direction while fixing the broken one.** Several
  "it worked this morning and now it doesn't" cycles came from changes to
  shared code — the playback path, the engine pump, device resolution — made in
  pursuit of the *other* direction.
- **One change at a time, with a test.** The test suite runs in about a minute
  (`python -m pytest -q`) and has caught real bugs before they reached a call,
  including the last sentence of every utterance being cut off.
- **Read the whole protocol document before patching field by field.** The
  OpenAI realtime endpoint cost several rounds of guessing at event names when
  the server's own error message listed the three valid ones.
- One test, `test_playback_waits_for_a_cushion_before_the_first_sound`, uses
  real sleeps and can flake under load. Re-run before believing it.
