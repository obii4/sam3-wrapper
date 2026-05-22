"""
Minimal SAM3 wrapper: image + one or more text prompts -> masks, boxes, scores.

Usage (CLI):
    python segment.py <image_path> "<prompt>"                       # single concept
    python segment.py <image_path> "glasses, hat, shirt"            # comma-separated
    python segment.py <image_path> "<prompt>" --output-dir DIR --confidence 0.3

Usage (library):
    from segment import Sam3Wrapper
    wrapper = Sam3Wrapper()
    result  = wrapper.segment("img.jpg", "glasses")             # single
    results = wrapper.segment("img.jpg", ["glasses", "hat"])    # multi -> dict by prompt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Distinct colors for multi-prompt overlays (RGB, 0-255).
_COLORS = [
    (0, 255, 0),     # green
    (255, 0, 0),     # red
    (0, 128, 255),   # blue
    (255, 255, 0),   # yellow
    (255, 0, 255),   # magenta
    (0, 255, 255),   # cyan
    (255, 128, 0),   # orange
    (128, 0, 255),   # purple
]


class Sam3Wrapper:
    """Loads SAM3 once, then runs (image, text) -> segmentation on demand."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        print(f"Loading SAM3 on {self.device}...")
        # Pass bpe_path explicitly — pkg_resources lookup breaks on editable installs.
        bpe_path = str(
            Path(__file__).parent
            / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
        )
        self.processor = Sam3Processor(
            self.model, confidence_threshold=self.confidence_threshold
        )
        print("SAM3 loaded.")

    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: Union[str, List[str]],
    ) -> Union[Dict, Dict[str, Dict]]:
        """
        Run SAM3 on an image.

        - `prompt` as str:       returns one result dict.
        - `prompt` as list[str]: returns {prompt: result_dict} for each.

        Each result dict has:
            masks:  list of HxW bool numpy arrays
            boxes:  list of [x1, y1, x2, y2] numpy arrays
            scores: list of float confidence scores
            prompt: the text prompt used
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(prompt, str):
            return self._segment_one(image, prompt)

        # Multi-prompt: set image once, reset prompts between runs.
        state = self.processor.set_image(image)
        results = {}
        for p in prompt:
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(state=state, prompt=p)
            results[p] = {
                "masks": [m.cpu().numpy() for m in output["masks"]],
                "boxes": [b.cpu().numpy() for b in output["boxes"]],
                "scores": [float(s) for s in output["scores"]],
                "prompt": p,
            }
        return results

    def _segment_one(self, image: Image.Image, prompt: str) -> Dict:
        state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return {
            "masks": [m.cpu().numpy() for m in output["masks"]],
            "boxes": [b.cpu().numpy() for b in output["boxes"]],
            "scores": [float(s) for s in output["scores"]],
            "prompt": prompt,
        }


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.asarray(mask).squeeze().astype(bool)
    if m.shape == (h, w):
        return m
    from scipy.ndimage import zoom
    return zoom(
        m.astype(float),
        (h / m.shape[0], w / m.shape[1]),
        order=0,
    ) > 0.5


def save_visualization(
    image_path: Union[str, Path],
    results: Dict[str, Dict],
    out_path: Union[str, Path],
) -> None:
    """
    Save side-by-side: original | color-coded mask overlay | color-coded boxes.

    `results` is the dict-of-dicts returned by segment() when given a prompt list.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    image = Image.open(image_path).convert("RGB")
    img_arr = np.array(image)
    h, w = img_arr.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Overlay all masks, color-coded by prompt.
    overlay = img_arr.astype(float).copy()
    legend_patches = []
    for i, (p, result) in enumerate(results.items()):
        color = np.array(_COLORS[i % len(_COLORS)], dtype=float)
        prompt_mask = np.zeros((h, w), dtype=bool)
        for mask in result["masks"]:
            prompt_mask |= _resize_mask(mask, h, w)
        if prompt_mask.any():
            overlay[prompt_mask] = overlay[prompt_mask] * 0.5 + color * 0.5
        legend_patches.append(
            Patch(facecolor=color / 255.0, label=f'{p} (n={len(result["masks"])})')
        )

    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title("Masks")
    axes[1].axis("off")
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc="lower right", fontsize=8)

    # Boxes, color-coded by prompt.
    axes[2].imshow(image)
    for i, (p, result) in enumerate(results.items()):
        color = tuple(c / 255.0 for c in _COLORS[i % len(_COLORS)])
        for box, score in zip(result["boxes"], result["scores"]):
            x1, y1, x2, y2 = box
            axes[2].add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            ))
            axes[2].text(
                x1, y1 - 4, f"{p} {score:.2f}",
                color=color, fontsize=7,
            )
    axes[2].set_title("Detections")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _slug(text: str) -> str:
    """Filesystem-safe slug for prompt keys in npz."""
    return "".join(c if c.isalnum() else "_" for c in text.strip())[:40] or "prompt"


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 on an image with one or more text prompts."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "prompt", type=str,
        help='Text prompt(s). Comma-separated for multiple, e.g. "glasses, hat, shirt"',
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./results"),
        help="Directory for outputs (default: ./results)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Confidence threshold 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Also save binary masks to masks.npz (keys: <prompt>_<idx>)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    if not prompts:
        raise ValueError("No valid prompts after parsing.")

    out_dir = args.output_dir / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = Sam3Wrapper(confidence_threshold=args.confidence)
    results = wrapper.segment(args.image, prompts)

    print()
    for p, result in results.items():
        print(f'"{p}": {len(result["masks"])} instance(s)')
        for i, score in enumerate(result["scores"]):
            print(f"  [{i}] score={score:.3f}")

    save_visualization(args.image, results, out_dir / "overlay.png")

    w, h = Image.open(args.image).size
    summary = {
        "image": str(args.image),
        "image_size": {"height": h, "width": w},
        "confidence_threshold": args.confidence,
        "prompts": {
            p: {
                "num_detections": len(result["masks"]),
                "scores": result["scores"],
                "boxes": [b.tolist() for b in result["boxes"]],
            }
            for p, result in results.items()
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_masks:
        mask_arrays = {}
        for p, result in results.items():
            slug = _slug(p)
            for i, mask in enumerate(result["masks"]):
                mask_arrays[f"{slug}_{i:03d}"] = _resize_mask(mask, h, w)
        if mask_arrays:
            np.savez_compressed(out_dir / "masks.npz", **mask_arrays)

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
