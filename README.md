# SAM3 Wrapper

A minimal command-line wrapper around Meta's [SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything with Concepts). Pass an image and one or more text prompts; get back masks, bounding boxes, confidence scores, and a visualization.

```bash
python segment.py path/to/image.png "red car"
python segment.py path/to/image.png "glasses, hat, shirt"   # multiple concepts
```

## What you get

For each run, a directory `results/<image_name>/` is created containing:

- `overlay.png` — original image, mask overlay, and labeled bounding boxes (side-by-side)
- `results.json` — per-prompt metadata (scores, boxes, counts)
- `masks.npz` — *only if `--save-masks` is passed.* Compressed binary masks, keyed `<prompt_slug>_NNN`.

Load masks with:

```python
import numpy as np
masks = np.load("results/<image_name>/masks.npz")
print(masks.files)          # ['glasses_000', 'hat_000', ...]
binary = masks["glasses_000"]   # bool HxW array
```

## Requirements

- Linux (tested) or macOS
- Python 3.12+
- NVIDIA GPU with CUDA 12.6+ (CPU works but is very slow)
- A Hugging Face account with access to [`facebook/sam3`](https://huggingface.co/facebook/sam3) (see step 3)

## Installation

### 1. Create a conda environment

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
```

### 2. Install PyTorch (GPU build)

```bash
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
```

If you don't have a GPU, install the CPU build instead:

```bash
pip install torch==2.7.0 torchvision torchaudio
```

### 3. Get Hugging Face access to SAM 3

SAM 3 weights are gated. You must do this before running the wrapper or model loading will fail.

1. Create a Hugging Face account at https://huggingface.co/join (if you don't already have one).
2. Visit the model page: https://huggingface.co/facebook/sam3
3. Click **"Request access"** and accept the license. Approval is usually automatic but can take a few minutes.
4. Create a read-only access token at https://huggingface.co/settings/tokens (click **"Create new token"**, give it `read` permissions).
5. Authenticate from the terminal:

   ```bash
   pip install huggingface_hub
   hf auth login
   ```

   Paste the token when prompted. (Older versions of the CLI use `huggingface-cli login` — either works.)

   To verify, run:

   ```bash
   hf auth whoami
   ```

### 4. Install wrapper + SAM 3 dependencies

```bash
pip install -r requirements.txt
```

This installs everything SAM 3 needs at runtime — including a handful of packages (`einops`, `pycocotools`, `decord`, ...) that SAM 3's own `pyproject.toml` under-declares as dependencies. Without these you'll hit cascading `ModuleNotFoundError`s.

### 5. Clone and install SAM 3

From this directory (`sam3_wrapper/`):

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
cd ..
```

## Usage

### Command line

```bash
python segment.py <image_path> "<text_prompt>" [options]
```

**Arguments:**

| Argument | Description |
| --- | --- |
| `image_path` | Path to the input image (PNG, JPG, etc.) |
| `text_prompt` | Open-vocabulary phrase, or comma-separated list of phrases (e.g. `"glasses, hat, shirt"`). Each phrase is run independently and color-coded in the overlay. |
| `--output-dir DIR` | Where to write results (default: `./results`) |
| `--confidence FLOAT` | Score threshold 0.0–1.0 (default: `0.3`) |
| `--save-masks` | Also write `masks.npz` (compressed binary masks, keyed `<prompt>_NNN`) |

**Example** (using the bundled sample image):

```bash
python segment.py images/DSC_1132.jpg "glasses" --confidence 0.4
```

The first run downloads SAM 3 weights (~several GB) from Hugging Face and caches them under `~/.cache/huggingface/`. Subsequent runs load from cache.

### As a Python library

```python
from segment import Sam3Wrapper

wrapper = Sam3Wrapper(confidence_threshold=0.3)

# Single prompt -> one result dict
result = wrapper.segment("images/DSC_1132.jpg", "glasses")
print(f"Found {len(result['masks'])} instances")

# Multiple prompts -> {prompt: result_dict}
results = wrapper.segment("images/DSC_1132.jpg", ["glasses", "hat"])
for prompt, r in results.items():
    print(prompt, len(r["masks"]))
```

Each result dict has `masks` (list of `numpy` boolean arrays), `boxes` (list of `[x1, y1, x2, y2]`), and `scores` (list of floats).

Load the model once and reuse `wrapper.segment(...)` across many images — initialization is the expensive part.

## Prompting tips

Specific phrases work much better than generic ones. `"red sedan"` will beat `"car"`; `"thin carbon fiber"` will beat `"line"`. If you get no detections, try:

1. Lowering `--confidence` (e.g. `0.2`).
2. Rewording the prompt with material, color, or shape descriptors.
3. Checking the prompt matches what's visually distinctive in the image.

## Troubleshooting

**`401 Unauthorized` or `gated repo` error when loading weights**
You haven't been granted access to `facebook/sam3` yet, or you haven't run `hf auth login`. Re-do step 3.

**`CUDA out of memory`**
Use a smaller image, or run on CPU by setting `device="cpu"` when constructing `Sam3Wrapper`.

**`ModuleNotFoundError: No module named 'pkg_resources'`**
`setuptools` 81+ dropped `pkg_resources`, which SAM 3 still imports. Downgrade:
```bash
pip install "setuptools<81"
```

**`ModuleNotFoundError: No module named 'einops'` / `'pycocotools'` / `'decord'` / `'cv2'`**
You skipped `pip install -r requirements.txt`. SAM 3 under-declares these in its own `pyproject.toml`, so the wrapper's `requirements.txt` pins them explicitly. Run it.

**`ModuleNotFoundError: No module named 'sam3'`**
You didn't run `pip install -e .` inside the cloned `sam3/` directory (step 4).

**Slow first run**
Normal — model weights are downloading. Watch `~/.cache/huggingface/` to confirm progress.

## Files

```
sam3_wrapper/
├── README.md          # This file
├── requirements.txt   # Wrapper dependencies (excludes torch + sam3)
├── segment.py         # CLI + Sam3Wrapper class
└── sam3/              # Cloned from facebookresearch/sam3 (after step 4)
```
