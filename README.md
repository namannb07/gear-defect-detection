---
title: Gear Defect Detection
emoji: ⚙️
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app/streamlit_app.py
python_version: 3.11
pinned: false
license: mit
short_description: Zero-shot gear defect detection with AnomalyCLIP
---

# Gear Defect Detection

Zero-shot industrial gear defect detection using
[AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) (ICLR 2024). Upload an
image or use your webcam and get an anomaly heatmap, score, and PASS/FAIL
verdict — no gear-specific training data, no fine-tuning required.

## How it works

AnomalyCLIP learns *object-agnostic* "normal" and "abnormal" text prompts on
an auxiliary dataset (MVTec AD), then transfers zero-shot to any new object
class. The shipped 22 MB checkpoint already encodes those prompts, so you
just point it at a gear photo and it produces:

- a per-pixel anomaly heatmap (CLIP patch features compared to the learned
  prompts at multiple ViT layers, gaussian-smoothed),
- an image-level anomaly score in [0, 1],
- a PASS / FAIL verdict against a configurable threshold.

The CLIP backbone is `ViT-L/14@336px`. Inference resolution is 518×518.

## Requirements

- Python 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management
- ~1 GB free disk for the CLIP weights (`~/.cache/clip/`)
- Optional NVIDIA GPU + matching CUDA toolkit (auto-detected at runtime;
  falls back to CPU otherwise)

Tested on Linux (including NVIDIA Jetson Orin / aarch64). Should also work on
macOS and Windows via the equivalent `uv run …` commands listed below.

## Quick start

```bash
bash scripts/setup.sh
uv run streamlit run app/streamlit_app.py
```

`setup.sh` runs `uv sync`, clones the upstream AnomalyCLIP source into
`vendor/AnomalyCLIP/`, patches a hardcoded path in the upstream loader,
copies the pretrained checkpoint to `models/anomalyclip.pth`, and warms the
CLIP weights cache (~890 MB download on first run).

## Project structure

```
gear-defect-detection/
├── app/
│   ├── inference.py          # predict() + CLI
│   ├── webcam_inference.py   # OpenCV live loop (q to quit)
│   ├── streamlit_app.py      # dashboard
│   ├── model.py              # cached AnomalyCLIP loader
│   └── utils.py              # device, paths, heatmap helpers
├── vendor/AnomalyCLIP/       # populated by setup.sh
├── datasets/                 # placeholder folders (zero-shot — not strictly required)
├── models/anomalyclip.pth    # pretrained checkpoint (22 MB)
├── outputs/                  # auto-saved composites from CLI runs
├── samples/                  # drop your own test images here
└── scripts/                  # setup/infer/webcam/train helpers
```

## Usage

### Streamlit dashboard

```bash
uv run streamlit run app/streamlit_app.py
```

Two tabs:
- **Upload image** — pick a JPG/PNG and run inference; verdict pill, score
  bar, three-pane visualization (original / heatmap / overlay), and a
  download button for the composite PNG.
- **Webcam snapshot** — browser-side camera capture (one-shot). For
  continuous live video, use `bash scripts/webcam.sh`.

A sidebar slider sets the PASS/FAIL threshold; a device badge shows whether
inference is running on GPU or CPU.

### Single-image CLI

```bash
uv run python -m app.inference --image samples/my_gear.jpg
# or
bash scripts/infer.sh samples/my_gear.jpg --threshold 0.55
```

Saves `outputs/<imagename>_result.png` (a labeled three-pane composite) and
prints the score + verdict.

### Live webcam

```bash
bash scripts/webcam.sh                 # default camera 0
bash scripts/webcam.sh --camera 1      # alternate camera
bash scripts/webcam.sh --threshold 0.6 # adjust PASS/FAIL cutoff
```

Opens a native OpenCV window with the heatmap overlay and a colored
PASS/FAIL banner. Press **q** to quit. Inference runs in a background thread
so the preview stays smooth even if a single inference takes seconds on CPU.

### Windows / non-bash users

The shell scripts are convenience wrappers. The underlying commands work
anywhere:

```powershell
uv sync
uv run python -m app.inference --image samples\my_gear.jpg
uv run python -m app.webcam_inference
uv run streamlit run app/streamlit_app.py
```

You'll still need to manually clone `https://github.com/zqhang/AnomalyCLIP`
into `vendor/AnomalyCLIP/` and copy
`vendor/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth` to
`models/anomalyclip.pth` if you skip `setup.sh`.

## GPU / CPU notes

Device is auto-detected per run via `torch.cuda.is_available()`:

- On Linux/Windows machines with a CUDA-compatible NVIDIA GPU and matching
  driver, inference runs on GPU (~ a few hundred ms per frame).
- On CPU (including Jetson, since the PyPI ARM torch wheel is CPU-only),
  inference takes a few seconds per frame for ViT-L. The webcam loop runs
  inference in a background thread so the live window stays responsive.

If you have a Jetson and want CUDA, you'll need NVIDIA's Jetson-specific
torch wheels (Python 3.10 only) — outside the scope of this portable setup.

## Using your own gear images

AnomalyCLIP needs **no** training data to detect anomalies on your gears —
just point it at the image. For a curated demo, drop a few photos into
`samples/` and reference them from the CLI or upload them in the dashboard.

If you later want to *fine-tune* the prompt embeddings on a custom auxiliary
dataset, see `vendor/AnomalyCLIP/train.sh` and the upstream README.

## Troubleshooting

- **"Model checkpoint not found" / "AnomalyCLIP source not found"** — run
  `bash scripts/setup.sh` once.
- **First run is slow** — the CLIP ViT-L/14 backbone (~890 MB) downloads
  once into `~/.cache/clip/`; subsequent runs are fast.
- **CLIP download fails** — manually fetch
  `https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt`
  into `~/.cache/clip/`.
- **`torch.cuda.is_available()` is False on a GPU machine** — your installed
  torch CUDA build doesn't match your driver. Install the matching
  `torch`/`torchvision` from `pytorch.org`.
- **`No webcam detected`** — confirm your camera works elsewhere (`ls
  /dev/video*` on Linux), or pass `--camera N` to select a different index.
- **Streamlit runs but inference is sluggish on CPU** — expected. Lower the
  Streamlit usage to single uploads (one inference at a time) or run on a
  GPU machine.

## Deploy to Hugging Face Spaces

The repo is also configured for one-click deployment on
[Hugging Face Spaces](https://huggingface.co/spaces) using the YAML frontmatter
at the top of this README plus the same files used for Streamlit Cloud:

| File | Purpose |
|---|---|
| `README.md` (frontmatter) | tells HF this is a Streamlit Space, points at `app/streamlit_app.py`, pins Python 3.11 |
| `requirements.txt` | pip-installable deps with CPU-only torch wheels |
| `packages.txt` | apt deps (`libgl1`, `libglib2.0-0`) for opencv |
| `.gitattributes` | tracks `*.pth` via git-lfs (HF recommends LFS for files >10 MB) |
| `vendor/AnomalyCLIP/` | committed source so import works without `setup.sh` |
| `models/anomalyclip.pth` | committed 22 MB pretrained checkpoint (LFS) |

### Steps

1. Install [git-lfs](https://git-lfs.com/) once on your machine
   (`sudo apt install git-lfs && git lfs install`).
2. Create a new Space at
   [huggingface.co/new-space](https://huggingface.co/new-space):
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free, 16 GB RAM — plenty for AnomalyCLIP + CLIP ViT-L)
   - **Visibility**: public or private, your choice
3. Push this directory to the Space's git remote:
   ```bash
   cd /home/quasi/Desktop/gear-defect-detection
   git init
   git lfs install
   git lfs track "*.pth"
   git add -A
   git commit -m "Initial gear defect detection app"
   git branch -M main
   git remote add origin https://huggingface.co/spaces/<your-user>/<your-space>
   git push -u origin main
   ```
   You'll be prompted for your HF username and an
   [access token](https://huggingface.co/settings/tokens) (use a token with
   `write` scope as the password).
4. The Space will start building automatically. The first launch installs deps
   (~3 min) and downloads CLIP ViT-L/14 (~890 MB, another ~3 min) — total cold
   start ~6–8 min. The sidebar will show `Environment: Hugging Face Spaces`
   once running.

### HF Spaces vs Streamlit Cloud

- **Memory**: HF's free CPU basic tier provides 16 GB RAM vs Streamlit Cloud's
  ~1 GB. AnomalyCLIP + CLIP ViT-L runs in roughly 1.5–2 GB, so OOM kills are
  not a concern on HF.
- **Cold start**: roughly comparable on CPU (deps install + CLIP weight pull).
- **Persistent storage**: neither tier has it on the free plan; the CLIP
  backbone re-downloads on every cold start.
- **GPU**: HF offers paid GPU hardware tiers (T4, L4, A10G, A100) that you can
  toggle in the Space settings. To use one, swap the CPU torch wheels in
  `requirements.txt` for the matching CUDA wheels from
  [pytorch.org](https://pytorch.org/get-started/previous-versions/).
- **Continuous webcam mode** (`scripts/webcam.sh`, native OpenCV window) is
  local-only on either platform — use the **Webcam snapshot** tab in the cloud.

## Deploy to Streamlit Community Cloud

The repo is configured for one-click deployment at
[share.streamlit.io](https://share.streamlit.io/) using these committed
files:

| File | Purpose |
|---|---|
| `requirements.txt` | pip-installable deps with CPU-only torch wheels |
| `packages.txt` | apt deps (`libgl1`, `libglib2.0-0`) for opencv |
| `runtime.txt` | pins Python 3.11 |
| `.streamlit/config.toml` | server + theme settings |
| `vendor/AnomalyCLIP/` | committed source (path-patched) so import works without `setup.sh` |
| `models/anomalyclip.pth` | committed 22 MB pretrained checkpoint |

### Steps

1. Push this directory to a public GitHub repo:
   ```bash
   cd /home/quasi/Desktop/gear-defect-detection
   git add -A
   git commit -m "Initial gear defect detection app"
   git branch -M main
   git remote add origin git@github.com:<your-user>/<your-repo>.git
   git push -u origin main
   ```
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with your
   GitHub account.
3. Click **New app** → pick the repo, branch `main`, main file path
   `app/streamlit_app.py`. Leave Python version on the default (3.11).
4. Click **Deploy**. The first launch installs deps (~3 min) and downloads
   CLIP ViT-L/14 (~890 MB, another ~3 min) — total cold start ~6–8 min. The
   sidebar will show `Environment: Streamlit Cloud` once running.

### Resource caveats

- Streamlit Community Cloud's free tier has ~1 GB RAM. AnomalyCLIP +
  CLIP ViT-L runs in roughly 1.5–2 GB at inference time. **Expect occasional
  OOM kills on the free tier.** If your use case is more than demos, run on
  a paid tier or self-host.
- The CLIP backbone (~890 MB) re-downloads on every cold start since
  Streamlit's container disk is ephemeral between deploys.
- Continuous webcam mode (`scripts/webcam.sh`, native OpenCV window) is
  local-only and not available in the cloud — use the **Webcam snapshot**
  tab there.

## License

This project is MIT. Upstream [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)
is also MIT-licensed.
