r"""Stage-2 cleanup: drop residue blobs that sit outside the layer's plausible region.

Facial parts (nose/mouth/eyes/brows/ears/headwear...) can only live inside the head
region of the character; leftover blobs elsewhere on the canvas are diffusion
residue from the 4-step LoRA and get removed.

  python clean_layers2.py --src_dir <dir> --out_dir <dir> [--preset soft] [--dilate 40]
"""

# Portable paths: everything is resolved relative to this file / the repo root.
import os
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))          # tools/fast_decomp -> repo root
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import argparse, json, os, os.path as osp
import numpy as np
from PIL import Image
from scipy import ndimage

PRESETS = {
    "soft": dict(pixel_thresh=64,  min_component=48,  min_layer_pixels=300, max_alpha_blank=150),
    "mid":  dict(pixel_thresh=96,  min_component=96,  min_layer_pixels=400, max_alpha_blank=150),
}
FACIAL = ["headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash",
          "eyewear", "ears", "earwear", "nose", "mouth", "eyes"]
HEAD_REF = ["head", "front hair", "back hair", "face"]


def alpha_of(path):
    return np.asarray(Image.open(path).convert("RGBA"))[..., 3]


def clean(arr, p, allowed=None):
    out = arr.copy(); a = out[..., 3]
    peak = int(a.max())
    if peak < p["max_alpha_blank"]:
        return np.zeros_like(out), dict(blanked=True, reason="no_peak", peak_alpha=peak, removed_outside=0)
    strong = a >= p["pixel_thresh"]
    if not strong.any():
        return np.zeros_like(out), dict(blanked=True, reason="empty_after_thresh", peak_alpha=peak, removed_outside=0)
    lab, n = ndimage.label(strong, structure=np.ones((3, 3), np.int32))
    keep = np.zeros_like(strong)
    removed_outside = 0
    if n:
        for i in range(1, n + 1):
            m = lab == i
            size = int(m.sum())
            if size < p["min_component"]:
                continue
            if allowed is not None and not (m & allowed).any():
                removed_outside += size      # blob lies outside the plausible region
                continue
            keep |= m
    kept = int(keep.sum())
    if kept < p["min_layer_pixels"]:
        return np.zeros_like(out), dict(blanked=True, reason="too_little_left", peak_alpha=peak,
                                       removed_outside=removed_outside)
    a[~keep] = 0
    return out, dict(blanked=False, reason="", peak_alpha=peak, kept_px=kept,
                     removed_outside=removed_outside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--preset", default="soft", choices=list(PRESETS))
    ap.add_argument("--dilate", type=int, default=40)
    ap.add_argument("--pixel_thresh", type=int, default=None)
    args = ap.parse_args()
    p = dict(PRESETS[args.preset])
    if args.pixel_thresh is not None:
        p["pixel_thresh"] = args.pixel_thresh

    head = None
    for ref in HEAD_REF:
        fp = osp.join(args.src_dir, ref + ".png")
        if osp.exists(fp):
            m = alpha_of(fp) > 25
            head = m if head is None else (head | m)
    if head is not None and args.dilate > 0:
        head = ndimage.binary_dilation(head, iterations=args.dilate)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for f in sorted(os.listdir(args.src_dir)):
        if not f.endswith(".png"):
            continue
        stem, arr = f[:-4], np.asarray(Image.open(osp.join(args.src_dir, f)).convert("RGBA"))
        if stem.startswith("src_"):
            Image.fromarray(arr).save(osp.join(args.out_dir, f)); continue
        allowed = head if (head is not None and stem in FACIAL) else None
        cleaned, info = clean(arr, p, allowed)
        Image.fromarray(cleaned).save(osp.join(args.out_dir, f))
        report[stem] = dict(before=round(float((arr[..., 3] > 25).mean()), 4),
                            after=round(float((cleaned[..., 3] > 25).mean()), 4), **info)
    with open(osp.join(args.out_dir, "_clean_report.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(params=p, dilate=args.dilate, head_available=head is not None, layers=report),
                  fh, indent=2, ensure_ascii=False)
    print(f"preset={args.preset} params={p} dilate={args.dilate} head_ref={head is not None}")
    print(f"blanked: {[k for k,v in report.items() if v['blanked']]}")
    print(f"{'tag':<12}{'before':>8}{'after':>8}{'blank':>7}{'outside_px':>12}")


if __name__ == "__main__":
    main()
