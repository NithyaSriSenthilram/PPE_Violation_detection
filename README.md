# SentinelVision AI

AI-powered CCTV surveillance and site-safety monitoring. Detects people, tracks
them with stable identities, checks PPE compliance, and raises incidents for
restricted-area intrusion, loitering, abnormal movement, possible falls and
crowd anomalies — with snapshot and video evidence attached to every one.

Built to be honest about itself: `GET /api/diagnostics` reports the inference
backend that is *actually* running, the verbatim reason every other backend is
not, whether the Mojo kernels are genuinely in use (with the benchmark that
decided each one), and whether PPE state comes from a trained model or a
clearly-labelled colour heuristic.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Installation](#installation)
- [Running it](#running-it)
- [Environment configuration](#environment-configuration)
- [Model setup](#model-setup)
- [Camera configuration](#camera-configuration)
- [Zones](#zones)
- [Video upload and analysis](#video-upload-and-analysis)
- [MAX setup](#max-setup)
- [Mojo setup](#mojo-setup)
- [API reference](#api-reference)
- [Testing](#testing)
- [Performance](#performance)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## What it does

| Capability | How it works | Status |
|---|---|---|
| **Person detection** | YOLOv8 via ONNX Runtime, hardware-accelerated where available | Production |
| **Person tracking** | ByteTrack + Kalman filter, persistent IDs through occlusion | Production |
| **PPE — helmet / vest** | Trained YOLO11 PPE detector (`person`, `helmet`, `no-helmet`, `vest`, `no-vest`), each item associated to a specific tracked person | Production |
| **PPE — future items** | `gloves`, `boots`, `goggles`, `mask`, `harness` are already in the taxonomy and rules; they need a model that emits those classes | Ready, model-dependent |
| **Restricted-area intrusion** | Polygon zones, tested against each person's foot point | Production |
| **Loitering** | Per-person, per-zone dwell time measured in wall-clock seconds | Production |
| **Abnormal movement** | Speed in body-heights/second, so one threshold works at any depth | Production |
| **Possible fall** | Upright → rapid vertical drop → sustained horizontal posture | Advisory (AI inference) |
| **Crowd anomaly** | Absolute head count plus deviation from a rolling baseline | Production |
| **Incident generation** | Central event engine with severity, debounce and a burst cap | Production |
| **Evidence** | Timestamped snapshot + a clip spanning the moments around the event | Production |
| **Real-time alerts** | One WebSocket carrying detections, events, camera status, job progress | Production |
| **Historical search** | Filter by camera, type, severity, status, person, date, free text | Production |
| **Camera management** | RTSP, HTTP/MJPEG, local webcam, or file replay | Production |
| **Video analysis** | Upload a recording, get incidents and an annotated copy back | Production |
| **Analytics** | Trend, category, severity and per-camera breakdowns | Production |

Adding a detection category means writing one `Analyser` subclass and one row in
the severity map. The pipeline, event engine, API and dashboard need no changes.

---

## Architecture

```
 RTSP / Webcam / HTTP / Uploaded file
                │
                ▼
   ┌───────────────────────────┐   Per-camera worker thread.
   │  VideoSource              │   Live sources keep only the newest frame
   │  (backend/video/source)   │   (dropping, not queueing, so the view
   └───────────┬───────────────┘   stays current); file sources read every
               │                   frame and can loop.
               ▼
   ┌───────────────────────────┐   Letterbox → RGB CHW float32.
   │  Preprocessing            │   The conversion is a Mojo kernel where
   │  (inference/preprocess)   │   that measured faster than numpy.
   └───────────┬───────────────┘
               ▼
   ┌───────────────────────────┐   Backend chosen once at startup:
   │  Detector  (abstraction)  │   max → onnx → ultralytics → mock.
   │  ├─ ONNX Runtime  ← default   Callers never know which one ran.
   │  ├─ Modular MAX
   │  ├─ Ultralytics
   │  └─ Mock (synthetic)
   └───────────┬───────────────┘
               ▼
   ┌───────────────────────────┐   Persistent person IDs. Every event is
   │  ByteTracker              │   keyed to one, so identity churn would
   │  (backend/tracking)       │   break incident history.
   └───────────┬───────────────┘
               ▼
   ┌───────────────────────────┐   PPE · Intrusion · Loitering · Movement
   │  Analysers  (pure)        │   Fall · Crowd. No DB, no files, no
   │  (backend/analysis)       │   alerts — just event candidates.
   └───────────┬───────────────┘
               ▼
   ┌───────────────────────────┐   Confidence gate → severity → debounce
   │  Event engine             │   → burst cap → evidence → persist →
   │  (backend/events/engine)  │   broadcast.
   └───────┬───────────┬───────┘
           │           │
           ▼           ▼
   ┌────────────┐  ┌──────────────┐
   │  SQLite    │  │  Event bus   │  Thread-safe, bounded queues. A slow
   │  (→ PgSQL) │  │              │  browser tab never throttles capture.
   └─────┬──────┘  └──────┬───────┘
         │                │
         ▼                ▼
   ┌─────────────────────────────┐
   │  FastAPI  +  WebSocket      │
   └─────────────┬───────────────┘
                 ▼
   ┌─────────────────────────────┐   Detections arrive as normalised
   │  React dashboard            │   coordinates and are drawn as SVG over
   │  (frontend/)                │   the MJPEG frame — crisp at any size,
   └─────────────────────────────┘   layers toggleable, no re-encoding.
```

### Layout

```
backend/
├── main.py              FastAPI app factory + startup/shutdown ordering
├── config.py            every tunable, from the environment
├── logging_conf.py      structured logs with an ALERT level and throttling
├── schemas.py           the API contract (Pydantic)
├── api/                 routers: cameras, zones, events, statistics,
│                        videos, evidence, health/diagnostics, websocket
├── db/                  SQLAlchemy models, engine, repository queries
├── video/               source abstraction, per-camera pipeline, manager,
│                        rolling clip buffer, overlay renderer
├── inference/           detector interface, backends, pre/post-processing,
│   ├── backends/        onnx · max · ultralytics · mock
│   ├── ppe.py           PPE model detector, association, heuristic fallback
│   ├── ppe_taxonomy.py  PPE items, class synonyms, PPE_CLASS_MAP parsing
│   └── mojo/            kernels.mojo, build.sh, ctypes bridge + fallbacks
├── tracking/            ByteTrack, Kalman filter, durable track summaries
├── analysis/            one module per behaviour rule
├── events/              taxonomy, event engine, evidence writer, event bus
└── jobs/                uploaded-video analysis runner

frontend/src/
├── components/          app shell, camera tile, detection overlay, charts,
│                        watch band, alert row, incident modal, zone editor
├── pages/               overview, live wall, alerts, analytics, analysis,
│                        cameras & zones, diagnostics
├── hooks/               shared WebSocket, polling with tab-visibility pause
├── lib/                 typed API client, formatting
└── styles/              design tokens + global primitives
```

---

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | Python 3.11+ | Threads for blocking capture, asyncio for the API |
| Video | OpenCV (headless) | Decoding, encoding, colour conversion, drawing |
| API | FastAPI + Pydantic v2 | Typed contract, generated OpenAPI, native WebSocket |
| Inference | ONNX Runtime | CoreML / CUDA / TensorRT providers from one build |
| Tracking | ByteTrack + Kalman | Uses low-confidence detections to survive occlusion |
| Numeric kernels | Mojo (optional) | Compiled inner loops; used only where measured faster |
| Accelerated inference | Modular MAX (optional) | Real backend, activated only when it can load the model |
| Database | SQLAlchemy 2 → SQLite | PostgreSQL needs only a `DATABASE_URL` change |
| Frontend | React 18 + Vite + TypeScript | No UI framework; ~90 KB gzipped total |

Deliberate omissions: no charting library (the SVG charts are ~400 lines and
match the design system exactly), no scipy (the Hungarian solver is included
and exact), no state-management library, no CSS framework.

---

## Installation

**Requirements**: Python 3.11+, Node 18+, ~2 GB disk (mostly the one-off model
export). A GPU is optional. Nothing here requires NVIDIA hardware.

```bash
git clone <repository-url> sentinelvision
cd sentinelvision
./scripts/setup.sh              # add --with-mojo for the accelerated kernels
```

The script creates `.venv`, installs dependencies, copies `.env.example` to
`.env`, and exports a detector. Every optional step degrades cleanly.

<details>
<summary>Manual installation</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

pip install ultralytics            # needed only to export the model
python scripts/fetch_models.py     # -> models/yolov8n.onnx

cd frontend && npm install && cd ..
```
</details>

---

## Running it

```bash
make backend     # API  → http://127.0.0.1:8008  (docs at /api/docs)
make frontend    # UI   → http://localhost:5173
```

The dev server proxies `/api` and `/ws`, so the app runs same-origin and there
is no CORS to configure locally. Point it elsewhere with
`VITE_API_TARGET=http://host:port`.

> **Port note** — the default is **8008**, not 8000, because 8000 is so often
> already in use. Change `APP_PORT` in `.env` and `VITE_API_TARGET` together.

First thing to check:

```bash
make diagnostics
```

That tells you the truth about the deployment before you trust a single
detection.

### Production

```bash
cd frontend && npm run build          # → frontend/dist, serve as static files
APP_ENV=production .venv/bin/python -m uvicorn backend.main:app \
  --host 0.0.0.0 --port 8008 --workers 1
```

Use **one** worker. Camera pipelines are in-process threads; multiple workers
would each open every camera and multiply the inference load. Scale by running
one instance per group of cameras behind a reverse proxy, sharing a PostgreSQL
database.

Setting `APP_ENV=production` stops exception detail from reaching clients.

### Deploying to Render

`render.yaml` describes the whole deployment as a Blueprint:

| Service | Type | Settings |
|---|---|---|
| `ppe-detection-api` | Docker web service | `Dockerfile` at repo root, health check `/api/health`, plan **Free** (512 MB RAM, no persistent disk, sleeps after 15 min idle) |
| `ppe-detection-frontend` | Static site | root `frontend`, build `npm ci && npm run build`, publish `dist`, SPA rewrite `/* → /index.html` |

The image installs FFmpeg, OpenCV and ONNX Runtime, copies the committed
detector models (`yolov8n.onnx`, `ppe.onnx`, `hardhat_prompts.npz`) and
downloads the hard-hat validator from its public Hugging Face source at build
time. The image defaults to the **q4f16** build of the validator (53 MB, same
measured accuracy as fp32): the fp32 build alone costs ~585 MB of RSS once
ONNX Runtime has optimised it, which is more than the whole free instance.
With q4f16, `INFERENCE_THREADS=1` and `ONNX_CPU_MEM_ARENA=false` the full
stack measures ~270 MB with models loaded, ~400 MB peak during a 720p
analysis and ~465 MB at 1080p, one job at a time. MAX/Mojo are not in the
image; `/api/diagnostics` reports ONNX Runtime as the active backend.

Backend environment (set in `render.yaml`):

```ini
PORT=10000                 # injected by Render; PORT beats APP_PORT
APP_ENV=production
DATA_ROOT=/var/data        # uploads/, processed/, evidence/, db/surveillance.db
INFERENCE_BACKEND=onnx
MOJO_ENABLED=off
INFERENCE_THREADS=1        # free tier: one shared CPU
ONNX_CPU_MEM_ARENA=false   # return scratch buffers between runs (512 MB cap)
ANNOTATED_FFMPEG_THREADS=1 # ffmpeg finalise shares the 512 MB; 1 thread halves its peak
ANNOTATED_MAX_WIDTH=1280   # frames scaled to this width after decode; caps per-frame memory
MAX_CONCURRENT_JOBS=1      # one analysis at a time; further uploads queue
JOB_MAX_RSS_MB=470         # fail the job cleanly before the 512 MB kill
JOB_MAX_SECONDS=7200       # wall-clock limit per job
AUTO_START_CAMERAS=false
MAX_UPLOAD_MB=200
CORS_ORIGINS=              # prompted at Blueprint creation: the static site URL
API_KEY=                   # optional shared secret
```

Frontend environment (build-time): `VITE_API_URL=https://<your-api>.onrender.com`.
Empty `VITE_API_URL` means same-origin, which is what the Vite dev proxy and
a reverse-proxy deployment use.

Everything the server writes lives under `DATA_ROOT`. On the free plan that
is ephemeral container storage: uploads, renders, evidence and the SQLite
database are wiped on every deploy, restart and sleep/wake cycle, so download
results while the service is up. Mounting a persistent disk at `/var/data`
(paid plans) makes them durable with no other change. Analysis jobs that were
mid-flight when a process died are marked `FAILED` at the next startup
("Interrupted by a server restart") rather than staying `RUNNING`.

Steps: push to GitHub → Render Dashboard → *New → Blueprint* → select the
repo → Render prompts for `CORS_ORIGINS` and `VITE_API_URL` (both are
`sync: false`, never hard-coded) → enter the two services' public URLs →
apply. Changing `VITE_API_URL` later needs a frontend rebuild.

---

## Environment configuration

Everything lives in `.env` (see `.env.example` for the annotated full list).
No thresholds or paths are hard-coded anywhere.

```ini
APP_ENV=development
APP_PORT=8008
DATABASE_URL=sqlite:///./data/surveillance.db

INFERENCE_BACKEND=auto            # auto | max | onnx | ultralytics | mock
MODEL_PATH=./models/yolov8n.onnx
CONFIDENCE_THRESHOLD=0.35
INFERENCE_WIDTH=640

TARGET_FPS=12                     # pipeline throttle
DETECT_EVERY_N_FRAMES=2           # detector cadence; tracker fills the gaps
MAX_STREAM_WIDTH=960              # decode-side downscale cap

LOITERING_THRESHOLD=60            # seconds
RUNNING_THRESHOLD=2.2             # body-heights per second
CROWD_THRESHOLD=15                # people in a region

EVENT_COOLDOWN_SECONDS=30         # per (camera, type, person, zone)
EVENT_MAX_PER_MINUTE=12           # burst cap per (camera, type, zone)
CLIP_BUFFER_SECONDS=8             # rolling pre-event footage

PROCESSED_DIR=./processed         # full annotated renders of uploads
EVENT_OVERLAY_SECONDS=2.5         # how long an event banner stays on screen
ANNOTATED_MAX_WIDTH=1920          # cap; annotations scale with the frame
ANNOTATED_FFMPEG_FINALISE=true    # faststart + audio mux + transcode
SHOW_PPE_BOXES=true               # tight boxes: helmet on head, vest on torso
SHOW_PERSON_BOXES=false           # the whole-person rectangle
SHOW_TRACK_ID=false               # small "ID #010" under each PPE label

PPE_CONFIDENCE_THRESHOLD=0.35     # helmet/vest — asserted present
PPE_ABSENCE_CONFIDENCE_THRESHOLD=0.25  # no-helmet/no-vest — scores lower
PPE_INTAKE_CONFIDENCE_THRESHOLD=0.20   # model floor; below both of the above
REQUIRED_PPE=helmet,vest          # helmet,vest,gloves,boots,goggles,mask,harness
PPE_ALLOW_HEURISTIC=false         # true → colour fallback when no model
MOJO_ENABLED=auto                 # auto | on | off
API_KEY=                          # empty disables auth
```

### Tuning notes

- **`RUNNING_THRESHOLD` is in body-heights per second**, not pixels. That is
  what makes one value work for someone near the lens and someone at the end of
  a corridor. Pixel thresholds have to be retuned per camera and still misfire
  with depth.
- **`EVENT_MAX_PER_MINUTE` is the backstop, not the main defence.** The
  per-person cooldown handles a continuous violation. The burst cap catches the
  case where person *identity* is unstable — a crowded doorway or heavy
  occlusion churns track IDs, and a new ID means a fresh cooldown key. Without
  the cap that is exactly how an operator gets hundreds of near-identical
  alerts. Suppressed counts are reported, never silently dropped.
- **`DETECT_EVERY_N_FRAMES=2`** roughly doubles throughput at negligible cost
  for pedestrian-speed motion. Set it to 1 for fast-moving subjects.

### PostgreSQL

```ini
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/sentinel
```

```bash
pip install "psycopg[binary]"
```

No code changes. There is no SQLite-specific column type or raw SQL in the
codebase, and a test asserts that stays true.

---

## Model setup

```bash
python scripts/fetch_models.py                  # yolov8n → models/yolov8n.onnx
python scripts/fetch_models.py --model yolo11n   # newer, slightly better
python scripts/fetch_models.py --imgsz 512       # smaller input, faster
```

Ultralytics is used **only** for this export. The serving path is ONNX Runtime
and never imports torch.

Any YOLOv8/v11-family ONNX export works; point `MODEL_PATH` at it and put a
matching newline-delimited `.names` file beside it.

### PPE model

PPE compliance runs on a **trained object-detection model**, not a colour test.
`make models` fetches and exports it:

```bash
python scripts/fetch_models.py --ppe     # -> models/ppe.onnx + models/ppe.names
```

The default is a YOLO11s fine-tuned on a construction-safety dataset, with five
classes: `helmet`, `no-helmet`, `vest`, `no-vest`, `person`. The explicit
*absence* classes matter — a model that only labels helmets can report a missing
one by inference alone, which is a weaker signal and is scored as such.

It loads through the same backend chain as the person detector
(`backend/inference/registry.build_detector`), so MAX is tried first, ONNX
Runtime with CoreML is the working path on Apple Silicon, and the Mojo decode
and NMS kernels are shared. There is no second copy of the inference code, and
no mock fallback: a PPE model that cannot load produces no PPE readings at all,
and diagnostics says exactly why.

```ini
PPE_MODEL_PATH=./models/ppe.onnx
PPE_MODEL_LABELS_PATH=./models/ppe.names
PPE_CONFIDENCE_THRESHOLD=0.35    # independent of CONFIDENCE_THRESHOLD
PPE_ABSENCE_CONFIDENCE_THRESHOLD=0.25   # the absence classes need their own bar
PPE_INTAKE_CONFIDENCE_THRESHOLD=0.20    # what the model itself decodes at
REQUIRED_PPE=helmet,vest         # what a violation is, per site
PPE_ALLOW_HEURISTIC=false        # see "Heuristic fallback" below
```

#### Class mapping

Classes are resolved **by name**, never by index, through the taxonomy in
`backend/inference/ppe_taxonomy.py`. Common naming conventions are handled out
of the box — `helmet` / `hardhat` / `hard_hat` / `with_helmet`, and
`no_helmet` / `NO-Hardhat` / `head` / `bare_head` for the absence — as are the
vest, glove, boot, goggle, mask and harness equivalents. Anything unrecognised
is *reported*, not guessed at, and mapped explicitly:

```ini
PPE_CLASS_MAP=Hard_hat=helmet:present,No-Helmet=helmet:absent,Worker=person
```

Adding a new category is `REQUIRED_PPE=helmet,vest,gloves` plus a model with a
glove class. The rules, event metadata, association geometry and UI all read
from the taxonomy.

#### Association

Detecting a helmet *somewhere* in the frame says nothing about whether a
particular person is wearing one, so each PPE box is attributed to a specific
tracked person by two signals together:

- **containment** — the fraction of the PPE box inside the person box. IoU is
  the wrong measure: a helmet is a tiny fraction of a body, so its IoU with a
  whole person stays low however certainly it is worn.
- **body band** — where the item sits vertically within the person box against
  where that item belongs. A helmet at waist height, or a vest box overlapping
  the person standing behind its wearer, is rejected.

`PPE_ASSOCIATION_THRESHOLD` (default 0.55) sets the containment floor.

#### Tuning the PPE thresholds

The PPE model is judged against **three** bars, not one, because presence and
absence are different questions and it does not score them alike. Measured over
the reference footage:

```text
class        n    p25    median   max     below 0.50
vest        102   0.43    0.77    0.86       30%
helmet       87   0.32    0.56    0.81       41%
no-vest      34   0.23    0.46    0.81       59%
no-helmet    37   0.22    0.34    0.54       97%
```

Read the last row carefully: `no-helmet` **never once exceeds 0.54**. Under a
single shared 0.50 bar, 97% of that class is discarded and the class is
effectively switched off — a bare-headed worker reads as `UNKNOWN` forever
rather than as missing a helmet. The presence classes suffer a milder version
of the same thing: a saturated *orange* hi-vis vest scores 0.37 where a
yellow-green one scores 0.84, because PPE training sets are dominated by
yellow-green, so a 0.50 bar drops real vests off real people.

Hence:

| Setting | Default | Applies to |
|---|---|---|
| `PPE_CONFIDENCE_THRESHOLD` | 0.35 | `helmet`, `vest`, `person` — an item asserted **present** |
| `PPE_ABSENCE_CONFIDENCE_THRESHOLD` | 0.25 | `no-helmet`, `no-vest` — an item asserted **missing** |
| `PPE_INTAKE_CONFIDENCE_THRESHOLD` | 0.20 | the floor the model decodes at |

The intake floor must stay **below both** of the others. Filtering happens in
the decode path, which sees only class indices; the semantic bars are applied
one layer up, in the only place that knows whether a class means present or
absent. An intake floor above either bar silently caps it.

Lowering the absence bar does not make the system trigger-happy, because the
score is not what gates a violation — `PPE_MIN_CONSECUTIVE_FRAMES` (default 5)
is. A finding has to hold across consecutive frames before it becomes an event,
so a single weak reading raises nothing. Helmets carry even less risk at a low
bar, because every one of them is re-checked by the validator described next.

To measure your own model rather than inheriting these numbers, run it over
your footage with `PPE_INTAKE_CONFIDENCE_THRESHOLD=0.05` and look at the score
distribution per class. Set each bar under the bulk of its own class, not under
the loudest one.

#### Hard-hat validation — a cap is not a helmet

A PPE model is trained to find head *coverings*. Ask one what a worker in a
baseball cap is wearing and it says `helmet`, and it says it with confidence:

```text
PPE detector on 45 real head crops from the site footage
    27 of 28 baseball caps  ->  "helmet",  0.52 - 0.80 confidence
```

That is not a threshold problem. The detector is confident and wrong, so
raising `PPE_CONFIDENCE_THRESHOLD` only discards the real hard hats alongside
the caps. Only an industrial hard hat is valid PPE, so **every helmet the
detector proposes is checked by a second stage that asks a different
question**, and that answer — not the detection — decides compliance:

```text
person detector -> tracking -> PPE detector -> helmet 0.76
                                                    |
                                          HARD-HAT VALIDATOR
                                          "is this an industrial hard hat?"
                                              /              \
                                    hard hat 0.94        cap 0.82
                                          |                  |
                                       HELMET            NO HELMET
                                                         MISSING_HELMET
```

**What it is.** CLIP ViT-B/32 run zero-shot over a bank of concrete headwear
concepts — *a hard hat*, *an industrial safety helmet*, *a baseball cap*, *a
knitted beanie*, *a bare head* — whose probabilities are grouped into hard-hat
and not-hard-hat. It is not fine-tuned for the task, and it is chosen on
evidence: every public "hardhat detection" model is trained on the same
helmet/head datasets and inherits the identical cap failure, while a
vision-language model genuinely holds the distinction. Install it with
`python scripts/fetch_models.py --hardhat` (351 MB, or `--hardhat-variant
q4f16` for 53 MB at the same measured accuracy). It runs on ONNX Runtime like
everything else; CoreML is deliberately not used for it, measured at 110 ms
per crop against 55 ms on CPU for this ViT.

**Measured**, on 45 head crops from the reference footage — 16 industrial hard
hats, 29 caps and bare heads, every one of which the PPE detector called a
helmet:

| | cap false-positive rate | hard-hat recall |
|---|---|---|
| PPE detector alone | **96%** | 100% |
| + hard-hat validator @ 0.50 | **0%** | 75% |

Reproduce it with `make hardhat-metrics` (`--sweep` for the threshold curve,
`--crops` for your own labelled set). `cap_false_positive_rate` is the number
the stage exists to drive to zero, and the threshold is set where it measures
best: at 0.70 recall falls to 31% for no gain in precision.

**The error direction is deliberate.** A hard hat this stage fails to recognise
costs an operator one glance at the footage. A cap it accepts costs them the
finding entirely, and reports a bare-headed worker as compliant. So *unknown is
never compliance*:

| Situation | Helmet verdict | Source |
|---|---|---|
| Validator confirms a hard hat | `HELMET`, at the *validator's* confidence | `detected` |
| Validator says cap / beanie / bare head | `NO HELMET` | `rejected` |
| Head crop too small, validator crashed, no validator installed | `NO HELMET` | `unverified` |
| `HARDHAT_VALIDATION_ENABLED=false` | the detector's verdict stands | unchanged |

The last two rows are different on purpose. *Unavailable* means the stage
should have run and could not, so it certifies nothing and diagnostics report
the deployment as **not compliance-ready**; *disabled* is an operator choosing
to trust the detector, logged loudly at startup and named on every rendered
frame as `HARD HAT UNVALIDATED`.

The two confidences stay separate everywhere — API, event metadata, evidence
and the burned-in label. A rejected helmet records `detector_confidence: 0.76`
beside `validation: not_hard_hat, validation_confidence: 0.82`, and the label
reads `NO HELMET 82%`: the detector's 76% measured that something was on the
head, which is not evidence that it was PPE.

Validation runs after association, on the head box already attributed to that
person, so tracking and association are untouched. Each person is re-checked
every `HARDHAT_VALIDATE_EVERY_N_DETECTIONS` frames (the check costs ~55 ms) and
the verdict is the mean of the last `HARDHAT_SMOOTHING_WINDOW` checks, so one
bad look cannot flip a worker's status — with the existing
`PPE_MIN_CONSECUTIVE_FRAMES` debounce and the event cooldown still behind it.

`HARDHAT_ALLOW_HEURISTIC=true` permits a documented structural fallback (shell
smoothness, dome curvature, silhouette symmetry, aspect — never colour, since
hard hats and caps share every colour). It is off by default, capped at 0.75
confidence and reported as `HEURISTIC` everywhere, because a fallback that
wrongly passes a cap reproduces the exact failure this stage exists to stop.

#### Heuristic fallback

When no model is installed, `PPE_ALLOW_HEURISTIC=true` re-enables the HSV
colour/region estimator. It is a real measurement but materially less reliable:
confidence is capped at 0.62, every result is tagged `method="heuristic"`, and
the dashboard labels it **HEURISTIC FALLBACK** in the alert row, the incident
detail, the overlay and the burned-in snapshot. With `PPE_ALLOW_HEURISTIC=false`
and no model, PPE is reported as *undetermined* and no violation can fire.

`make diagnostics` reports the model path, whether it exists, its format,
whether it loaded, the backend serving it, the resolved class map, the
thresholds and whether the heuristic is permitted — and prints an actionable
warning instead of crashing when the model is missing.

---

## Camera configuration

Add cameras in **Cameras & Zones**, or over the API:

```bash
# RTSP
curl -X POST localhost:8008/api/cameras -H 'Content-Type: application/json' -d '{
  "name": "Loading Bay 01",
  "location": "North Yard — Gate A",
  "source_type": "rtsp",
  "stream_url": "rtsp://user:pass@10.0.0.20:554/Streaming/Channels/101",
  "settings_json": {"profile": "thorough"}
}'

# Local webcam (device index)
... "source_type": "webcam", "stream_url": "0"

# File replay — a recording served as a live camera. Loops by default.
... "source_type": "file", "stream_url": "site-recording.mp4"
```

`file` sources resolve **inside `UPLOAD_DIR` only**; traversal attempts are
rejected. Set `settings_json.loop` to `false` for a single pass.

Detection profiles (`settings_json.profile`): `thorough`, `standard`,
`security_only`, `ppe_only`, `fast`.

An unreachable camera is **not** an error that stops anything. It is marked
`error` with the decoder's own message, retried with exponential backoff
(3s → 60s), and every other camera keeps running.

---

## Zones

Zones are polygons stored in **normalised (0–1) coordinates**, so one definition
survives a resolution change and renders identically on a wall display and a
phone. Draw them over the camera's own live frame in the zone editor — placing a
boundary on an abstract rectangle never lines up with the real doorway.

| Type | Behaviour |
|---|---|
| `restricted` | Entry raises `RESTRICTED_AREA`; also accrues dwell time |
| `monitored` | Dwell time only, no entry alert |
| `loitering` | Dwell time only |
| `crowd` | Head count for density and baseline-deviation alerts |

Membership uses each person's **foot point** (bottom-centre of the box), not the
centroid: a floor-marked exclusion zone is about where someone is standing, and
a centroid test fires for anyone leaning across the line.

Zone edits reach a running camera within ~5 seconds — no restart.

---

## Video upload and analysis

Upload a recording and the platform runs the same detection chain a live
camera runs — detect, track, PPE, behaviour, events — and returns **the whole
video back, annotated**. That render is the primary output: a two-minute
upload produces a two-minute MP4 with PPE boxes, zones and event banners
drawn onto every frame, playable in the browser.

```text
upload → decode → detect → track → PPE → analyse → events
                                  ↓
                     annotate frame → write frame       (every frame, in order)
                                  ↓
                   finalise (H.264 + faststart + audio)
                                  ↓
                   GET /api/videos/jobs/{id}/result → player
```

Per-event snapshots and clips are still captured, but they are secondary
evidence. A still cannot show whether a box actually tracks the person it
claims to; the render can.

### The render

Written to `PROCESSED_DIR` (default `./processed/`), named after the upload:

```text
data/uploads/<job-id>.mp4                      original, never modified
processed/factory_a1b2c3d4_annotated.mp4       full annotated render
```

Guaranteed properties, each covered by a test in `tests/test_video_output.py`:

| Property | How |
|---|---|
| One output frame per source frame, in order | Every decoded frame is annotated and written, including frames between detections — those reuse the current tracks, so the render never freezes or drifts |
| Original duration, fps and resolution | Taken from the source; the render is not resampled |
| Plays in Chrome and Safari | H.264 / `yuv420p`, verified by reopening the file and by ffprobe |
| Seeking without a full download | `-movflags +faststart` moves the MP4 index to the front, and the API answers Range requests with `206 Partial Content` |
| Audio preserved | The source track is muxed back in with ffmpeg — `cv2.VideoWriter` has no audio concept |
| Flat memory | Frames stream through one at a time; nothing accumulates |

`ANNOTATED_MAX_WIDTH` (default 1920) is the working width: a wider source is
scaled down once, straight after decode, and detection, tracking, PPE,
annotation and the render all run on that frame. A 4K upload therefore costs
the same memory as a 1920-wide one and does not produce an unplayable
multi-gigabyte file. Annotations are drawn on the working frame, so they stay
correctly positioned.

### What the render draws

The boxes are **PPE boxes, not people**. A rectangle around a whole body
carries one bit — *someone is here* — and at any distance it swallows the head
and torso the finding actually concerns. So each item is drawn where it lives:

```text
   ┌──────────────┐                      ┌─────────────────────────────┐
   │ NO HELMET 91%│                      │ PERSON #010                 │
   └──────────────┘                      │                             │
      ┌────────┐            instead of   │            (person)         │
      │        │  ← the head             │                             │
      └────────┘                         │                             │
   ┌──────────────┐                      │  Helmet:NO Vest:NO          │
   │ NO VEST 86%  │                      └─────────────────────────────┘
   └──────────────┘
   ┌────────────┐
   │            │  ← the torso
   └────────────┘
```

Person detection and tracking are untouched by this — they still drive PPE
association, zones, loitering, movement and every event. The person box is
simply no longer the thing on screen.

| Setting | Default | Draws |
|---|---|---|
| `SHOW_PPE_BOXES` | `true` | The tight per-item boxes and their labels |
| `SHOW_PERSON_BOXES` | `false` | The whole-person rectangle, ID and track readout |
| `SHOW_TRACK_ID` | `false` | A small `ID #010` line under each PPE label |

Where each box comes from, in order of how much it can be trusted:

1. **The model's own box**, used verbatim when it is the size of the thing it
   names — which is the normal case. The installed model's helmet boxes run
   0.09–0.15 of person height and its vest / no-vest boxes 0.38–0.55.
2. **Constrained to the body region** when the model answers with something
   person-sized, as some weights do for their absence classes. The box keeps
   its horizontal placement and is intersected with the region, so a "no vest"
   lands on a torso rather than a body.
3. **The body region itself** when there is no box at all — an absence deduced
   from omission, or a heuristic estimate. These are drawn **dashed**, so a
   derived box never passes for a detected one.

Regions are data, in `PPE_BODY_REGIONS` / `BODY_REGIONS`
(`backend/inference/ppe_taxonomy.py`): head, face, torso, hands and feet, each
a rectangle in fractions of the person box. Gloves, boots, goggles, masks and
harnesses are already mapped — a new item needs an entry there, not new
drawing code.

Labels are placed to stay readable: above the box by preference, then below,
then inside its top edge, and never over another label, the HUD or an event
banner — every rendered label is remembered and the next one steers around it.

`SHOW_PERSON_BOXES=true` restores the old readout. It is worth turning on for a
deployment whose events are mostly zone- and movement-based (intrusion,
loitering, falls), where the person marker *is* the finding — with it off,
those events are carried by the banner and the zone overlay alone.

If the render cannot be produced, the job is marked **FAILED** with the real
reason. A job never reports `COMPLETED` over a video nobody can open.

### Without ffmpeg

The render still works. Three things are given up, and each is reported as a
warning on the result rather than hidden:

- audio cannot be preserved (`Audio preservation unavailable`),
- the MP4 index stays at the end, so the browser downloads the whole file
  before it can seek,
- an unplayable codec cannot be transcoded.

### Zones on uploaded footage

Uploaded video has no camera, so no zone geometry exists and the zone rules
(intrusion, per-zone loitering) correctly find nothing. Pass `camera_id` with
the upload to borrow that camera's zones — they are then drawn on the render
and the rules fire against them:

```bash
curl -X POST http://localhost:8008/api/videos/upload \
  -F file=@factory.mp4 -F profile=thorough -F camera_id=<camera-id>
```

### Result API

```text
GET  /api/videos/jobs/{id}/result     URLs, playback metadata, event timeline
GET  /api/videos/jobs/{id}/video      stream the render (Range-capable, inline)
GET  /api/videos/jobs/{id}/download   download the render
GET  /api/videos/jobs/{id}/original   download the original upload
```

`result` returns API URLs, never filesystem paths, and reports what the
encoder actually produced:

```json
{
  "annotated_ready": true,
  "video_url": "/api/videos/jobs/a41e53ba.../video",
  "duration_seconds": 84.0, "fps": 15.0, "frames": 1260,
  "resolution": "960x540", "codec": "h264",
  "has_audio": true, "browser_compatible": true,
  "warnings": [],
  "timeline": [
    {"label": "Restricted Area Intrusion", "video_timestamp": 0.667, "severity": "HIGH"}
  ]
}
```

`timeline` entries carry `video_timestamp` — the seek target. Clicking one in
the dashboard jumps the player to the frame that produced the event.

### Detection profiles

Chosen per upload; each one runs a different set of analysers, trading coverage
for speed.

| Profile | Analysers | Use for |
|---|---|---|
| `fast` | intrusion, crowd | A quick pass over long footage |
| `standard` | PPE, intrusion, loitering, movement, crowd | The default |
| `thorough` | everything, including fall detection | Incident review |
| `ppe_only` | PPE | Compliance audits |
| `security_only` | intrusion, loitering, movement, fall, crowd | Security review without PPE |

`GET /api/videos/profiles` returns the live list.

---

## MAX setup

The MAX backend is **implemented for real** and activates automatically when it
can load the configured model.

```bash
pip install modular
```

Then restart and check `GET /api/diagnostics`.

### Measured result on the development machine

On an Apple M1 (macOS 26.6, arm64), MAX 26.5.0 installs and runs, and
`driver.accelerator_count()` reports the M1 GPU. But **MAX 26.5 cannot compile a
generic ONNX graph**:

```
RuntimeError: cannot compile input with format input has unknown contents
```

MAX now targets its own graph format plus safetensors/GGUF checkpoints rather
than acting as a drop-in ONNX runtime. So on this version there is no MAX path
for a YOLO ONNX model, and the platform says exactly that rather than pretending
otherwise:

```json
{ "name": "max", "available": false, "active": false,
  "reason": "MAX runtime v26.5.0 could not compile yolov8n.onnx: cannot compile input with format input has unknown contents" }
```

The backend probes the runtime, enumerates devices, attempts the load, and
records the engine's own error message. It will activate unchanged on a host or
MAX version where the load succeeds, or if `MODEL_PATH` points at a MAX-native
artefact.

**MAX is entirely optional.** ONNX Runtime with the CoreML provider is the
fastest real path on this hardware anyway — see [Performance](#performance).

---

## Mojo setup

Mojo is used for the numeric inner loops that run on every frame, and **only
where it measured faster than numpy on the host machine.**

```bash
python3.11 -m venv .venv-mojo
.venv-mojo/bin/pip install modular
./backend/inference/mojo/build.sh     # or: make mojo
```

The build emits a shared library exporting C-ABI symbols, loaded through
`ctypes`. The Python side never imports Mojo, and the compiled library loads
from a virtualenv that has no Modular installed. If the toolchain is missing,
the build script explains how to get it and exits 0 — the platform runs on the
numpy fallbacks and reports `mojo.active: false`.

### Self-calibration

At startup the bridge checks each kernel against its numpy twin for numerical
agreement, then benchmarks both, and enables Mojo **only where it wins**.
Shipping a "Mojo-accelerated" call that is slower than the fallback would be a
lie told in code, so the bridge measures instead of assuming.

Measured on the development machine (Apple M1):

| Kernel | Using | Mojo | numpy | Speedup |
|---|---|---|---|---|
| `nms` | **mojo** | 0.128 ms | 3.124 ms | **24.5×** |
| `points_in_polygon` | **mojo** | 0.005 ms | 0.018 ms | **3.70×** |
| `iou_matrix` | **mojo** | 0.009 ms | 0.024 ms | **2.53×** |
| `bgr_to_chw` | **mojo** | 0.284 ms | 0.505 ms | **1.77×** |
| `decode_yolov8` | numpy | 0.519 ms | 0.109 ms | 0.21× |

`decode_yolov8` stays on numpy: a vectorised `argmax` over an 80 × 8400 array
beats a scalar loop, and the calibration correctly refuses to use Mojo there.
Your numbers appear in `GET /api/diagnostics`.

Mojo is deliberately **not** used for model execution, orchestration, I/O or
anything else. `backend/inference/mojo/` is the whole boundary.

> **Mojo 1.0 note** — the language changed substantially: `fn` is gone (use
> `def`), the stdlib is `std.*`, origins are `MutAnyOrigin` / `ImmutAnyOrigin`,
> and `@export` rejects parametric functions, so pointer parameters must name a
> concrete origin. `kernels.mojo` documents the working forms.

---

## API reference

Interactive docs: **`/api/docs`** · OpenAPI: **`/api/openapi.json`**

### System
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness. `degraded` when running on synthetic detections |
| `GET` | `/api/diagnostics` | Real backend state, Mojo benchmarks, PPE method, config |

### Cameras
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/cameras` | List, with live telemetry merged in |
| `POST` | `/api/cameras` | Add (and start, if enabled) |
| `GET` | `/api/cameras/{id}` | One camera |
| `PUT` | `/api/cameras/{id}` | Update; restarts the pipeline when needed |
| `DELETE` | `/api/cameras/{id}` | Delete, cascading zones and events |
| `POST` | `/api/cameras/{id}/start` · `/stop` | Pipeline control |
| `GET` | `/api/cameras/{id}/runtime` | FPS, latency, tracks, frames dropped |
| `GET` | `/api/cameras/{id}/stream` | MJPEG preview (`?width=&fps=`) |

### Zones
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/zones` | Create (normalised polygon) |
| `GET` | `/api/zones/{camera_id}` | List a camera's zones |
| `PUT` | `/api/zones/detail/{zone_id}` | Update |
| `DELETE` | `/api/zones/detail/{zone_id}` | Delete |

### Events, alerts, evidence
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/events` | Search — `camera_id`, `event_type`, `severity`, `status`, `person_id`, `job_id`, `start`, `end`, `search`, `limit`, `offset` |
| `GET` | `/api/events/{id}` | One event |
| `PATCH` | `/api/events/{id}/status` | Acknowledge / resolve / dismiss, with notes |
| `GET` | `/api/alerts` | Active alerts, most severe first |
| `GET` | `/api/statistics` | KPIs, breakdowns, timeline (`?range=today\|24h\|7d\|30d\|custom`) |
| `GET` | `/api/evidence/{id}` | Manifest — distinguishes *pending* from *absent* |
| `GET` | `/api/evidence/{id}/snapshot` · `/clip` · `/export` | JPEG · MP4 · zip |

### Video analysis
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/videos/upload` | Upload and queue (multipart: `file`, `profile`, `start`) |
| `GET` | `/api/videos/jobs` · `/jobs/{id}` | Job list / progress |
| `POST` | `/api/videos/jobs/{id}/start` · `/cancel` | Re-run / cancel |
| `DELETE` | `/api/videos/jobs/{id}` | Delete job and its upload |
| `GET` | `/api/videos/profiles` | Profiles and the analysers each enables |

### WebSocket — `/ws`

```js
const ws = new WebSocket('ws://localhost:8008/ws?topics=detections,event')
```

Envelope: `{ type, payload, ts }`, where `type` is `hello`, `detections`,
`event`, `camera_status`, `job_progress` or `pong`. Narrow live with
`{"action":"subscribe","topics":[...],"cameras":[...]}`.

Detection boxes are **normalised 0–1**, so the client scales them to whatever
size it renders at. Subscriber queues are bounded: a browser tab that stops
reading has its stale messages dropped rather than throttling capture.

### Event schema

```json
{
  "event_id": "…", "event_type": "MISSING_HELMET", "label": "Missing Helmet",
  "camera_id": "…", "camera_name": "Loading Bay 01",
  "person_id": 12, "zone_id": "…", "zone_name": "Restricted Zone A",
  "severity": "HIGH", "confidence": 0.62, "status": "OPEN",
  "description": "Person #012 has no helmet (Helmet: NO | Safety Vest: YES)",
  "bbox": [141, 200, 255, 501],
  "snapshot_path": "evidence/snapshots/…jpg",
  "clip_path": "evidence/clips/…mp4",
  "detection_metadata": { "ppe_method": "model", "missing": ["vest"], "duplicates_suppressed": 0 },
  "timestamp": "2026-08-29T13:46:19.596250Z",
  "advisory": false
}
```

`advisory: true` on `POSSIBLE_FALL` and `ABNORMAL_MOVEMENT` marks results the
interface must present as AI inferences requiring human verification.

Auth is off when `API_KEY` is empty. When set, send `X-API-Key` on HTTP and
`?api_key=` on the WebSocket (browsers cannot set headers on the handshake).

---

## Testing

```bash
make test         # 321 tests, ~3min
make test-fast    # skips the slower real-file pipeline tests
make lint
make typecheck
```

Tests run against a temporary SQLite file and the **mock detector**, so no
model, GPU or network is needed — that is what the mock backend exists for. PPE
is covered the same way: association, class interpretation and the rules are
exercised with scripted detections rather than real weights, which is why
`ModelPPEDetector` takes an injected `Detector` rather than building its own.
The handful of tests that do want the real PPE model skip themselves when it is
not installed.

| Module | Covers |
|---|---|
| `test_config.py` | Path resolution, validation bounds, PostgreSQL passthrough |
| `test_tracking.py` | Hungarian optimality vs brute force, ID stability, occlusion recovery |
| `test_analysis.py` | Zone geometry, all five rules, PPE truth table, foot-point semantics |
| `test_ppe.py` | PPE class mapping and synonyms, person/PPE association, missing helmet / missing vest / compliant, confidence threshold, heuristic fallback, cooldown, duplicate suppression, evidence provenance, diagnostics |
| `test_events.py` | Severity, debounce, burst cap under ID churn, evidence containment |
| `test_api.py` | Every endpoint, filters, and the upload/traversal defences |
| `test_video.py` | Ring-buffer bounds, corrupt/missing sources, loop, real pipeline run |
| `test_video_output.py` | The annotated render over real footage: frame-for-frame output, duration/fps/resolution, decodability, browser codec, faststart, audio muxing, range requests, downloads, timeline seek targets, and failure handling |
| `test_annotation.py` | What the overlay draws: no person rectangle, PPE boxes on the body part they name, model boxes trusted or constrained, multiple people and multiple items, frame-edge clipping, label placement and collisions |
| `test_hardhat.py` | A cap never becomes a helmet: rejection by the validator, unknown/crashed/missing validators defaulting to NO HELMET, confidences kept separate, vest untouched, per-person association, temporal holding and smoothing, and the corrected verdict reaching the event engine. Plus the real validator over real head crops |
| `test_database.py` | Timezone round-trips, cascades, portability, upsert idempotence |

Several tests exist because they caught real bugs during development, and are
kept as regressions:

- **`scale_boxes` never clipped.** `np.clip(arr[:, [0, 2]], …, out=arr[:, [0, 2]])`
  writes into a *copy* — fancy indexing is not a view — so edge detections kept
  out-of-frame coordinates.
- **Evidence paths were written and resolved against different bases** (project
  root vs `evidence_root.parent`). It worked only while `EVIDENCE_DIR` sat
  directly under the project root, and broke every evidence download otherwise.
- **The YOLO head-orientation heuristic** ("transpose when rows > columns")
  mis-decoded any head with fewer anchors than attributes. It now resolves the
  layout from the label count.
- **`Subscription` was an unhashable dataclass** stored in a `set`, so every
  WebSocket connection raised `TypeError`.

---

## Performance

Measured on an Apple M1 (8 GB, macOS 26.6), 960 × 540 source, YOLOv8n at
640 × 640:

| Path | Latency | Throughput |
|---|---|---|
| ONNX Runtime + **CoreML** provider | **14.0 ms** | **71 fps** |
| ONNX Runtime, CPU provider | 37.4 ms | 27 fps |
| End-to-end pipeline (decode → detect → track → analyse → events) | 22–34 ms | ~12 fps at `TARGET_FPS=12` |
| Uploaded-video analysis, `ppe_only` profile | — | **1.6× real time** |

CoreML gives a **2.7×** speedup over CPU and is selected automatically.

### Per-stage breakdown

`make benchmark` measures every stage on real frames and prints what it
actually ran on. Measured over 150 frames of 960 × 540 footage, both models on
ONNX Runtime + CoreML, Mojo kernels active for `bgr_to_chw`, `iou_matrix`,
`points_in_polygon` and `nms`:

| Stage | Mean | p95 | Throughput |
|---|---|---|---|
| Person detection (YOLOv8n, 640²) | 16.9 ms | 20.9 ms | 59 fps |
| **PPE detection** (YOLO11s, 640², incl. association) | 30.4 ms | 36.9 ms | 33 fps |
| Tracking (ByteTrack + Kalman) | 0.27 ms | 0.51 ms | 3 700 fps |
| Event processing (6 analysers + engine) | 0.10 ms | 0.14 ms | 9 700 fps |
| **Full pipeline**, every stage on every frame | 47.7 ms | 56.8 ms | 21 fps |

Resident memory: ~107 MB for the interpreter, **+44 MB for both ONNX
sessions**, peaking around 330 MB with the benchmark's frame buffer (which the
live pipeline does not hold — it streams).

Reading these honestly:

- **PPE detection is now the dominant cost**, at roughly 1.8× the person
  detector. That is the price of a real model: YOLO11s is a substantially
  larger network than YOLOv8n. `PPE_EVERY_N_DETECTIONS` trades PPE latency for
  throughput when a camera needs the frame rate more than it needs per-frame
  PPE, and `DETECT_EVERY_N_FRAMES=2` already halves both detector costs on a
  live camera — which is why 21 fps here still sustains `TARGET_FPS=12`.
- **Tracking and event processing are free** at three to four orders of
  magnitude below the detectors. Optimising them would be pointless; the whole
  budget is in inference.
- These are numbers from an idle M1. Run `make benchmark` on your own hardware
  rather than trusting them — under other load on this same machine the same
  benchmark reported 2–4× worse figures, and it reports the backend and
  providers alongside the timings precisely so a slow result can be diagnosed
  rather than guessed at.

### What makes it fast

- **Execution providers in preference order** — TensorRT → CUDA → CoreML →
  ROCm → CPU. The provider actually granted is reported in diagnostics.
- **Detection cadence** (`DETECT_EVERY_N_FRAMES`) with the tracker filling the
  gaps.
- **Frames dropped, never queued** on live sources, so a slow analysis pass
  shows current footage instead of falling further behind.
- **Decode-side downscale cap** (`MAX_STREAM_WIDTH`) before any inference.
- **Mojo kernels** where they measured faster — NMS is 24.5× quicker.
- **JPEG-encoded clip buffer.** Raw frames would cost ~150 MB per camera for an
  8-second buffer; JPEG at quality 80 makes it ~4 MB. That decision is the
  difference between one camera and ten on a modest box.
- **Overlays drawn in the browser** from normalised coordinates — no server-side
  re-encoding, and layers toggle instantly.
- **One shared detector** across cameras; per-camera tracker, analysers and
  buffer.
- **Batched telemetry writes** (every ~3 s) instead of per-frame.

Rough capacity: an 8-core Apple Silicon machine handles **4–6 cameras** at
12 fps with `DETECT_EVERY_N_FRAMES=2`. Scale by lowering `TARGET_FPS`, raising
the detect interval, reducing `INFERENCE_WIDTH`, or running more instances.

---

## Security

- **Input validation** on every endpoint via Pydantic, including cross-field
  checks (`source_type` constrains `stream_url`).
- **Stream URLs** restricted to `rtsp`/`rtsps`/`http`/`https`. `file://` and
  friends are rejected, so a camera record cannot become a way to read local
  files. `file` sources resolve inside `UPLOAD_DIR` only.
- **Uploads**: extension allow-list, **server-generated** storage filename,
  streaming size limit, and a decoder probe. The client's filename is only a
  display label.
- **Evidence** is served by *event id*, never by a client-supplied path, and the
  resolved path is re-checked for containment inside `EVIDENCE_DIR` — so even a
  tampered database value cannot escape it.
- **CORS** is an explicit origin list, never a wildcard.
- **No secrets in source.** `API_KEY` comes from the environment and is compared
  in constant time.
- **Errors**: in production, clients get a generic message plus a request id;
  the full traceback goes to the server log only.
- **Containment**: a failing camera, analyser, PPE model or database write is
  caught, logged (throttled) and retried. One bad camera cannot take down the
  server.

**Not included** (deliberately, and you should add them before exposing this
beyond a trusted network): user accounts and RBAC, TLS termination, audit
logging of operator actions, rate limiting, and encryption of evidence at rest.

---

## Troubleshooting

**`GET /api/diagnostics` is the first stop for all of these.**

| Symptom | Cause and fix |
|---|---|
| `inference_backend: mock`, "Synthetic detections" banner | No real backend loaded. Check `backends[].reason` — usually a missing model: `python scripts/fetch_models.py` |
| Camera stuck `error` | Read `last_error`. Test the URL first: `ffprobe rtsp://…`. Check credentials, that the camera allows another stream, and reachability |
| RTSP takes ages to fail | It is capped at `STREAM_OPEN_TIMEOUT_SECONDS` (default 6 s). OpenCV's own default is ~30 s |
| PPE says `UNKNOWN` a lot | Undetermined is a real answer — below its bar the model says nothing rather than guessing, and undetermined never fires a violation. But check the bars first: presence and absence classes score very differently, and a single high threshold silently switches the absence classes off. Measure yours before adjusting, as described in *Tuning the PPE thresholds* |
| A bare-headed person reads as `UNKNOWN`, never as missing a helmet | The `no-helmet` class is scoring below `PPE_ABSENCE_CONFIDENCE_THRESHOLD`. Absence is the harder call and models score it well below the presence classes — on the reference footage `no-helmet` never exceeds 0.54. Lower that bar; `PPE_MIN_CONSECUTIVE_FRAMES` is what keeps it honest, not the score |
| Everyone flagged as missing a vest | Check `make diagnostics` — if `method` is `HEURISTIC FALLBACK` the model did not load and ordinary clothing is being colour-tested. Fix the model path |
| A helmet on a bench clears someone | It should not: association needs the box both contained in the person and in the right body band. Raise `PPE_ASSOCIATION_THRESHOLD` if it does |
| Hundreds of similar alerts | Person IDs are churning (crowding/occlusion). Lower `EVENT_MAX_PER_MINUTE`, raise `TRACK_MAX_AGE_FRAMES`, or improve camera placement |
| Low FPS | Raise `DETECT_EVERY_N_FRAMES`, lower `INFERENCE_WIDTH` to 512/416, lower `MAX_STREAM_WIDTH`, confirm an accelerated provider in diagnostics |
| Clip missing but snapshot present | Clips are written after a post-roll. The manifest distinguishes `pending` from absent; retry in a few seconds |
| `mojo.active: false` | Expected without the toolchain. Run `make mojo`. If the library loaded but nothing is active, numpy simply won — see the benchmark table |
| Port already in use | Default is 8008. Change `APP_PORT` **and** `VITE_API_TARGET` |
| Frontend shows "Cannot reach the backend" | Backend not running, or the proxy target is wrong for your `APP_PORT` |
| Video upload rejected | Extension not allow-listed, over `MAX_UPLOAD_MB`, or not decodable. The response says which |

---

## Known limitations

Stated plainly, because knowing these matters more than a longer feature list.

1. **The PPE model is a general construction-safety model, not yours.** It was
   trained on a public dataset, and its accuracy on your site, cameras, angles
   and lighting is unmeasured until you measure it. Helmet and vest are the
   only items it emits; `UNKNOWN` is a common and correct answer at distance or
   under occlusion, and an undetermined item never raises a violation. Fine-tune
   on site footage before treating it as a compliance record.
2. **Annotations are burned into the render, not overlaid by the player.**
   That is what makes a downloaded file still show them, and what lets the
   render stand as evidence — but it also means the overlay cannot be
   toggled off, restyled or re-rendered without re-running the analysis.
   The live dashboard draws its overlay in the browser precisely so that
   *is* toggleable; the two paths are deliberately different.
3. **The colour heuristic is a fallback, not a detector.** It only engages when
   no model is installed *and* `PPE_ALLOW_HEURISTIC=true`. HSV statistics over
   the head and torso are a real measurement, and they will be fooled by a
   yellow shirt, unusual lighting, or an odd camera angle. Confidence is capped
   at 0.62, every result is tagged `method="heuristic"`, and the UI labels it
   **HEURISTIC FALLBACK** everywhere it appears — never as an equal to a model
   reading.
4. **Fall detection is geometric inference.** Upright → rapid drop → sustained
   horizontal box. It will miss falls that end out of frame or behind an
   obstacle, and can fire on someone lying down deliberately. Labelled
   `POSSIBLE_FALL`, always advisory, never a medical determination.
5. **MAX cannot serve ONNX on version 26.5** — see [MAX setup](#max-setup). The
   backend is real and probes honestly; there is simply no path for this model
   format on that version.
6. **No re-identification.** Person IDs are per-camera and do not survive a long
   occlusion or a move between cameras. Someone leaving and returning gets a new
   ID. Cross-camera tracking needs a ReID model.
7. **Single-process camera pipelines.** One uvicorn worker owns all cameras.
   Scaling is horizontal (instances per camera group), not by adding workers.
8. **No authentication beyond an optional shared key.** No users, roles or audit
   trail. Do not expose this to an untrusted network as-is.
9. **Crowd counting is detection-based**, so it undercounts dense crowds where
   people occlude each other. Density-map methods are better past ~30 people.
10. **The detector is COCO-trained YOLOv8n** — the smallest model, chosen so the
   platform runs well on modest hardware. It misses small, distant and heavily
   occluded people. Use `yolov8s`/`yolov8m` where you have the headroom.
11. **Evidence is unencrypted on local disk** with no retention policy. Clips can
   fill a disk; add rotation before long-term deployment.
12. **The dashboard is dark-only.** A bright panel in a dim control room is an
    ergonomic problem, so a light theme is deliberately not offered.

---

## Roadmap

**Accuracy**
- Fine-tuning the PPE model on site-specific footage, and a labelled
  validation set so its accuracy here is measured rather than assumed
- Person re-identification for cross-camera and long-occlusion identity
- Pose estimation for genuinely reliable fall detection
- Density-map crowd counting for dense scenes

**Platform**
- User accounts, roles and an audit trail of operator actions
- Evidence retention policy with rotation and optional encryption at rest
- Multi-process camera sharding with a shared PostgreSQL backend
- Alembic migrations (the models are ready; only SQLite autocreate is wired)
- Notification sinks: email, webhook, Slack, SMS

**Detection**
- Abandoned-object and vehicle/ANPR categories
- Line-crossing with direction, and people counting in/out
- Configurable rule composition in the UI, without code
- Per-camera schedules (different thresholds by shift)

**Performance**
- TensorRT and OpenVINO backends alongside ONNX Runtime
- Batched multi-camera inference in one forward pass
- Hardware-accelerated decode (NVDEC / VideoToolbox)
- A MAX Graph implementation once it can consume this model class

---

## License and attribution

The detector is exported from **Ultralytics YOLOv8**, which is **AGPL-3.0**.
That licence affects redistribution of the weights and of derived work — review
it, or obtain an Ultralytics commercial licence, before shipping this
commercially. Swapping in a permissively-licensed ONNX detector requires only a
`MODEL_PATH` change.

Test fixtures are the standard Ultralytics sample images, used for verification
only.
