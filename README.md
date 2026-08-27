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
├── run.py                  <- start here
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

## Run it

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install PySide6 websockets aiohttp openai keyring

python run.py
```

**First run** shows only the setup window — devices from dropdowns, API keys
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

**After setup** you get the bubble and its panel: Start and Stop.

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

**152 tests pass:** `pytest`

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

- **The first-run wizard is not built.** Settings covers the same ground, but
  without the step-by-step verification the spec asks for (§11.7).
- **The debug view (Figure 12) is text-only**, via `format_debug_view`.
- **Backchannel suppression is implemented but not wired** into the outgoing
  pipeline — the S2ST model does its own endpointing, so it needs a different
  hook than the cascade path assumed.
- **`Ctrl+Alt+R` (repeat) is a stub.**
- **Device pinning is by name, not endpoint ID.** PortAudio does not expose
  the Windows endpoint ID; `pycaw` does. Phase 6.
- **Process loopback is deferred** (§5.2 / D7). Endpoint loopback on a virtual
  speaker is the v1 design.

## Next

Phase 0 on real hardware, then the evaluation set. Everything else waits on
those two.
