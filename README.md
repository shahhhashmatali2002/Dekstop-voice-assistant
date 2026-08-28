# Real-Time Urdu ↔ English AI Meeting Translator

Windows desktop application. Floating overlay. Python.
Design: `Urdu-English-Meeting-Translator-Technical-Spec.docx` (v2.0) — section
references throughout the code point back to it.

You speak Urdu; the client hears English. The client speaks English; you hear
Urdu. Teams is never modified and never knows we exist — we simply own the
audio endpoints it is pointed at.

---

## Layout

```
VoiceAssistant/
├── run.py                  <- meeting mode: two way, Teams
├── listen.py               <- listen mode: one way, any application, live
├── pretranslate.py         <- recorded video: prepared ahead, in sync
├── requirements.txt
├── app/
│   ├── models/             state, config, terminology, telemetry
│   ├── views/              overlay, transcript, settings, hotkeys, capture guard
│   ├── controllers/        main, engine, pipelines, VAD, segmenter, drift
│   ├── audio/              WASAPI devices, streams, ring buffer, conversion
│   └── providers/          speech-to-speech, recognition, translation, voices
├── tools/                  phase0, replay, bench_asr, ui_preview
└── tests/
```

`audio/` and `providers/` sit beside the MVC triad rather than inside it: they
are I/O adapters, and burying a WASAPI stream under `models/` would be filing
by convenience rather than by meaning. Controllers talk to them; views never do.

---

## Two modes, asked once

```powershell
python run.py --mode video      # something is playing; one cable, mic off
python run.py --mode meeting    # a two-way Teams call; two cables, mic live
```

Without `--mode` the first run asks and remembers the answer, and it is the
first thing on **Settings › Devices** afterwards — changing it there re-opens
the endpoints straight away, no relaunch. The dialog greys out the rows the
chosen mode does not use, so a microphone cannot be picked for a session with
no outgoing direction. The two
topologies are genuinely different and neither can be inferred:

| | video | meeting |
|---|---|---|
| capture | the cable the app is routed to | the cable Teams outputs to |
| play into | your speakers | your speakers |
| microphone | **not opened** | opened, translated to English |
| cables needed | one | two |
| directions | EN → UR | EN → UR and UR → EN |

Guessing wrong is not cosmetic: it opens a microphone nobody asked for, or
fails to open the one a call needs. In video mode the outgoing half of the
engine does not exist — no microphone stream, no second wire, no
speech-to-speech provider.

`listen.py` is the same topology as `--mode video` with its own small window
instead of the bubble.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install PySide6 websockets aiohttp openai keyring

python run.py
```

**First run** configures itself where it can: your microphone and headphones
come from whatever Windows is already using, and the two virtual cables are
matched against the virtual endpoints on the machine. If both are found it
saves and goes straight to the bubble. It only shows the setup window — devices from dropdowns, API keys
straight into Windows Credential Manager. Nothing else appears, because nothing
else can do anything yet. Cancel and the app exits.

Setup refuses any binding that would feed a device back into itself — pointing
the translator microphone or Teams' speaker at your real speakers creates a
loop that gets louder every pass, so it is rejected before a single stream
opens rather than after your laptop starts howling (§5.6 F1/F2).

Only **two** choices are actually required: the two virtual cables. Your
microphone and speakers default to whatever Windows is already using, and a
device you picked earlier that has since been unplugged falls back to the
default rather than blocking startup — it says so when it does.

**The bubble always comes up.** If audio cannot start it is red and the panel
says what is missing, with Settings inside it — no modal form in front of you.

```powershell
python run.py --allow-capture     # let every window appear in screenshots
```

Every window is excluded from screen capture by default (§2.4), which is why
a screenshot of the bubble or the transcript comes back without them.
`--allow-capture` waives it for one run and is never saved: a stored "off"
would sit there quietly until the call where it matters. `listen.py` takes the
same flag.

**On Start** the live transcript opens by itself, and closes again on Stop.

There is no file to edit by hand.

```powershell
python run.py --check         # what is installed, what is configured
python run.py --devices       # list audio endpoints
python run.py --ui-preview    # the overlay alone, no audio, no providers
```

The bubble and panel are frameless and drag themselves; they never take focus
from Teams. The transcript is an ordinary window with a real title bar, close
button and resize grips — it needs none of that overlay behaviour, and keeping
the frame means Windows provides all of it correctly. Escape closes it. Ctrl+C
in the console quits, as does Quit in the tray menu.

Failures are shown in a dialog with the full traceback behind "Show Details",
including ones raised on the engine thread — a desktop application should not
report its problems only to a console nobody is watching.

---

---

## Listen mode: hear anything in Urdu

`listen.py` is the second entry point and the one way round. No microphone, no
second cable, nothing that knows what Teams is. Whatever an application is
playing in English comes out of your headphones in Urdu — and the English does
not.

```
    YouTube, a Teams call, a recording, anything
        |
        |  Windows per-application output routing
        v
    CABLE Input            a virtual cable nobody is listening to
        |
        |  WASAPI loopback capture
        v
    [ passthrough wire ] --------------------------> headphones
        |                                            (before Start)
        +--> VAD -> English ASR -> segmentation
                 -> translation -> Urdu TTS --------> headphones
                                                     (after Start)
```

### One-time setup

1. Install **VB-CABLE** (vb-audio.com/Cable) and reboot.
2. Tell the tool which cable it is:

   ```powershell
   python listen.py --source "CABLE Input" --save
   ```

3. Route **only** the application you want translated:

   *Settings › System › Sound › Volume mixer › Chrome (or Teams) › Output
   device › CABLE Input*

Your headphones stay the Windows default, so every other application still
sounds exactly as it did. Nothing about the source application is modified and
it never knows we exist — the same principle as meeting mode, one endpoint
further out.

### Run it

```powershell
python listen.py                  # the window: Start, Pause, Subtitles
python listen.py --console        # no interface; starts at once, Ctrl+C quits
python listen.py --devices        # the playback endpoints you can route to
python listen.py --check          # install, devices, API keys
python listen.py --allow-capture  # let the windows appear in screenshots
```

Both windows are excluded from screen capture (§2.4), so a screenshot, the
Snipping Tool, OBS and a Teams share all see straight through them — the
window is simply absent, which is correct for a live transcript of the other
person's words and confusing the first time you try to photograph your own
tool. `--allow-capture` waives it for one run and is never saved.

Before Start, the cable is pumped straight to your headphones and you hear the
original English. Start mutes that wire and the Urdu takes over the same sink.
Stop, Pause, and every provider failure unmute it again: **you are never left
in silence** — §2.5, unchanged.

Exactly one thing reads the capture ring at a time, the wire or the pipeline,
because a ring read is destructive. That is the invariant `tests/test_listen.py`
spends most of its length on.

### Why the virtual cable is not optional

What we capture *is* the render stream of the endpoint we capture from, so
muting that endpoint mutes our own input along with it. There is no gain
setting that takes the English out of your ears and leaves it in ours.

The one mechanism that separates them is **process loopback** (§5.2 / D7):
capture a chosen process's audio whether or not it is being rendered anywhere.
PyAudioWPatch does not expose it, so until that is built the source application
has to render into a cable nobody hears.

### Fluency and lag are one dial

```powershell
python listen.py                     # fluent, the default for listen mode
python listen.py --preset balanced   # a compromise
python listen.py --preset fast       # conversation: short phrases, close to real time
```

A unit is emitted when the recogniser's text has been stable for
`stability_ms`, or when it hits `max_words`. Short units reach your ears sooner
and are synthesised in isolation, so the voice restarts its prosody every few
words — accurate, and it sounds like someone reading a list. Longer units are
spoken as whole thoughts, and arrive later. There is no setting that is both.

Three things were making the voice worse than the trade required, all now
fixed:

- **No jitter cushion.** The render callback drains the ring every period
  whether or not anything is filling it, and serves an empty ring as silence.
  Audio is now held for `playback_preroll_ms` before the first sample is
  released, so a late chunk lands in the cushion instead of punching a hole
  through the middle of a word.
- **Playback was serialised.** Synthesis waited for the previous utterance to
  finish, so the one to two seconds of recognising, translating and
  synthesising the next unit happened in the silence after it — a pause
  between every phrase. `_speak` now returns as soon as the first block is
  audible; the tail plays while the next unit is prepared.
- **The drift signal was the buffer, not the lag.** Queue depth was read from
  the sink ring, which the player deliberately keeps full, so it reported a
  permanent 500 ms of lag and time-stretched every utterance forever — a
  metallic edge on the voice at all times. It now counts audio still waiting
  to be written, which is the part that actually means falling behind.

A fourth is the one heard as **half-repeated sentences** — "or an advanced
student?" then "or an advanced student? Here are the rules." spoken again in
full. The gate reports end-of-speech at every 400 ms breath; the recogniser
does not, and keeps extending one hypothesis straight across it. Anything that
cleared "what we have already spoken" at those breaths — a `segmenter.reset()`
on speech onset, or a flush that ended the utterance — made the next
hypothesis arrive as though it were new, and it was spoken from the top with a
few words added. `_emitted` now survives a VAD pause and is compared word by
word, ignoring the punctuation the recogniser shuffles around, so a revision
contributes only what is genuinely new.

A fifth was worse than any of them: the segmenter kept `_emitted` across
utterances, so `_remainder` compared brand-new speech against an unrelated
sentence, found no shared prefix and returned nothing. The first utterance of
a session was translated and **every one after it was silently swallowed**.
`_emitted` is per-utterance state again; the last utterance's text is kept
separately, which is what actually stops a duplicated final being spoken
twice.

Making the queue-depth measurement honest also broke the drop threshold that
depended on it being dishonest. It could never report more than the sink
ring's 500 ms, so the 3 s "give up" threshold was unreachable and never fired.
Against real lag - a cascade genuinely runs three to six seconds behind - the
same number dropped every single unit and played nothing at all. The defaults
now match what they mean.

`--console` prints a health line every 15 seconds that tells them apart:
`stretch` above 1.00 is the metallic one, `dropped` is missing words, `gaps`
is a stutter inside a word.

### What to expect

Two to four seconds behind the speaker, dominated by endpointing (§9.3, tune
with `tools/replay.py --sweep stability`). On a video the picture runs ahead of
the Urdu; the subtitle window shows the English and the Urdu as each arrives so
you can follow the gap. In a live call that lag is the price of the cascade —
Urdu is not an available output language for any speech-to-speech model (§8.2),
so there is nothing faster to switch to.

---

## Pre-translated playback: Urdu in sync

```powershell
pip install youtube-transcript-api
python pretranslate.py "https://www.youtube.com/watch?v=..."
```

Live translation is always behind the picture, and that is a property of time
rather than a shortfall in the engineering: a sentence has to be spoken before
it can be translated, Urdu puts the verb where English does not, and then
recognition, translation and synthesis each take their turn. Three seconds is
the floor. Teams and Windows both ship *captions* rather than spoken
translation for the same reason.

**A recording has a future.** Its words already exist before you press play, so
this mode does not stream at all:

1. it reads the whole caption track and joins the fragments into sentences;
2. it translates every sentence with the rest of the video for context;
3. it synthesises all the Urdu;
4. **then** playback starts, each line fired at the timestamp it belongs to.

Nothing is being waited for during playback, so nothing is behind. The Urdu
lands where the English was. It also reads better than the live path, because
no sentence is translated as a fragment.

```powershell
python pretranslate.py URL --start 1:30     # resuming mid-video
python pretranslate.py URL --offset -0.4    # consistently late, nudge it
python pretranslate.py URL --out "Headset"  # where the Urdu is played
```

The original English is silenced exactly as in listen mode - point the browser
at the cable. The cable is only used to make the English inaudible here;
nothing is captured, so there is no loopback and no recogniser.

**What it cannot do:** anything live. A Teams call has no future to read ahead
into, so that stays on the live path and stays a few seconds behind. And a
video with captions disabled cannot be prepared this way — transcribing the
audio ahead of playback would work and is not built.

## What is built

| Layer | Verified? |
|---|---|
| `app/audio` primitives — ring buffer, conversion | tested, incl. threaded and tone fidelity |
| `app/controllers` — VAD, segmenter, drift | tested, incl. pitch preservation |
| `app/models` — state, telemetry, terminology | tested, incl. the audio-flowing invariant |
| `app/providers` | language guards, prompts, SSML tested — **network paths not exercised** |
| `app/audio` devices, streams, passthrough | ⚠️ **never executed — needs Windows** |
| `app/views` | renders correctly; focus and capture behaviour **need Windows** |
| `app/controllers/engine` + `main` | ⚠️ **never executed end to end** |
| `pretranslate.py` + `app/controllers/pretranslate` | tested: the merge, the context, the clock, the offset, the format conversion — **captions and providers are network paths, not exercised** |
| `listen.py` + `app/controllers/listen` | tested, incl. English-in / Urdu-out end to end against fake providers — **real devices and real providers not exercised** |

**301 tests pass:** `pytest`

---

## Start here: Phase 0, the audio spine

**Do not start with the AI.** If audio cannot travel reliably from Teams,
through this process, to your headphones — and from your microphone, through
this process, into Teams — then no model quality matters. Retire that risk
first; it takes a day.

Wear headphones for every step (§5.6 F1 — speakers create a feedback loop).

```powershell
python tools\phase0.py devices                       # 1. find endpoint names
python tools\phase0.py mic --mic "Headset"           # 2. level meter
python tools\phase0.py passthrough --mic "Headset" --out "Headset Earphone"
python tools\phase0.py loopback --target "CABLE Input"
python tools\phase0.py tone --virtual-mic "CABLE Input" --seconds 30
python tools\phase0.py wire --mic "Headset" --out "Headset Earphone" `
                            --virtual-mic "CABLE Input" --teams-speaker "CABLE-B Input"
python tools\phase0.py soak  ...same args...  --minutes 30
```

Every step ends with health counters. **`overflows` and `underruns` must both
be zero.** A non-zero `underruns` on the render path is the specific finding
Phase 0 exists to surface: it means the GIL discipline in §4.3 is not holding,
which is the known cost of Python (risk R4). The fix is to make the callbacks
do less — not to enlarge the buffers, which only hides it behind latency.

### Teams settings

| Setting | Value |
|---|---|
| Microphone | the virtual cable's **capture** half |
| Speaker | a second cable's render half, which we loopback-capture |
| Noise suppression | **Off** — §5.7, it will gate synthesised speech |

---

## Phase 1: measurement before providers

```powershell
python tools\replay.py sample.wav -v
python tools\replay.py sample.wav --sweep stability
python tools\bench_asr.py --list
python tools\bench_asr.py --provider deepgram --lang ur --keyterms
```

`replay.py` runs the whole VAD → stability → segmentation → latency chain
offline, with no devices and no GUI. The `--sweep` mode is the §9.3 tuning
exercise: endpointing dominates TTFTA, and it must be tuned against recordings
rather than in a live meeting. On the synthetic sample, stability at 100 ms
gives p50 1.92 s against 2.87 s at 350 ms — that is the trade you are making.

`bench_asr.py` is the one that settles the project's biggest risk. It needs
the evaluation set, which does not exist yet and is the **most valuable asset
in the project** (§15.2): 30–60 minutes of your real speaking style — real
technical content, real code-switching, real pace, on the real headset —
hand-transcribed. It reports two numbers:

- **overall WER** — how good the recogniser looks
- **technical-token WER** — whether the product actually works

The second is restricted to the English technical vocabulary embedded in your
Urdu. It is the number that predicts success and it is hidden inside the
first. R1 trips at roughly 15%.

---

## Hotkeys

Launches to the tray, brings the audio wire up in **passthrough**, and shows
the bubble in grey. Click it, press Start. Or use the hotkeys, which work
while Teams has focus:

| Hotkey | Action |
|---|---|
| `Ctrl+Alt+Space` | Pause / Resume — the time-critical one |
| `Ctrl+Alt+T` | Start / Stop |
| `Ctrl+Alt+H` | Hide the bubble |
| `Ctrl+Alt+R` | Repeat last utterance *(not implemented)* |

Working while Teams has focus is the whole point, and it also means another
application can already own one. `Ctrl+Alt+Space` is claimed by Teams' own
push-to-talk and by most input-method editors, so a conflict there is the
common case. When one is taken the app tries alternatives, binds the first
that works and tells you which — a silently moved hotkey is as useless as a
missing one. Set them explicitly in **Settings › Hotkeys** to stop them moving.

---

## The three rules this codebase is built around

**§4.3 — the callback does nothing.** A PortAudio callback holds the GIL while
it runs. If it is slow, the user hears a click. So it performs one pre-sized
numpy slice assignment and returns. No allocation, no logging, no locks, no
format conversion. If you are about to add a line to `_callback` in
`streams.py`, add it to the engine thread instead.

**§2.5 — passthrough is the resting state, not "off".** Teams is bound to our
virtual microphone permanently. An engine that only runs while translating
produces dead air whenever it is not, and the client hears silence without
knowing why. The wire starts with the application and stops with it, never
with the Start button. `Stop` ends translation, not audio. Every failure path
falls back to raw audio, never to silence.

**§2.4 — every window is excluded from screen capture.** An always-on-top
overlay appears in a screen share by default, showing the client a badge and a
live transcript of their own words. `ui/capture_guard.py` applies
`WDA_EXCLUDEFROMCAPTURE` to every top-level window. One window that misses it
defeats the whole measure. Verify from the far side of a real call, per build.

---

## Known gaps

- **The first-run wizard is not built.** Startup now auto-configures instead:
  microphone and headphones come from the Windows defaults, and the two cables
  are matched from the virtual endpoints present, saved, and the bubble opens
  straight away. What is missing is the step-by-step *verification* §11.7 asks
  for - it picks, it does not prove the picks work.
- **The debug view (Figure 12) is text-only**, via `format_debug_view`.
- **Backchannel suppression is implemented but not wired** into the outgoing
  pipeline — the S2ST model does its own endpointing, so it needs a different
  hook than the cascade path assumed.
- **`Ctrl+Alt+R` (repeat) is a stub.**
- **Device pinning is by name, not endpoint ID.** PortAudio does not expose
  the Windows endpoint ID; `pycaw` does. Phase 6.
- **Process loopback is deferred** (§5.2 / D7). Endpoint loopback on a virtual
  speaker is the v1 design, and it is what makes the virtual cable mandatory
  in listen mode.
- **`pipeline.original_mix_db` is still not implemented.** Mixing a little of
  the original underneath the Urdu needs a *summing* sink: the render sink is
  a ring buffer, so two writers interleave rather than add, and a second
  reader of the capture ring would starve the first. The setting is honoured
  nowhere, and listen mode deliberately does not pretend otherwise.

## Next

Phase 0 on real hardware, then the evaluation set. Everything else waits on
those two.

**Listen mode is the cheapest way to do the first one.** It needs one cable
instead of two, no microphone, and no outgoing pipeline, and it exercises the
same capture, ring, conversion and render code the meeting mode depends on.
If `python listen.py --console` translates a YouTube video into your
headphones without a click, the audio spine is sound and the remaining risk
is recognition quality — which is what `tools/bench_asr.py` and the
evaluation set are for.
