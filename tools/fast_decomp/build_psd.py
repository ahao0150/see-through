r"""Build a layered PSD without the Marigold depth model.

Pipeline: layer PNGs -> occlusion-based z-order (parallel, no GPU) -> synthetic
per-layer depth maps + info.json (same format the project expects) -> the repo's
own further_extr() PSD export.

The synthetic depth encodes z-order only: bottom layer gets the largest value,
because dump_parts_psd() sorts by depth_median descending and writes layers
bottom-first.

  python build_psd_algo.py --layer_dir <dir> [--workers 8] [--save_psd]
"""

# Portable paths: everything is resolved relative to this file / the repo root.
import json
import os
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))          # tools/fast_decomp -> repo root
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import argparse, json, os, os.path as osp, sys
import numpy as np
from PIL import Image

if REPO not in sys.path:
    sys.path.insert(0, REPO)
from zorder_occlusion import solve
from utils import inference_utils as iu


def write_algo_depth_and_psd(layer_dir, workers=8, min_margin=0.08, min_cov=0.15,
                             save_psd=True, verbose=True, patch_layers=False,
                             patch_dilate=6, patch_color_thresh=45, silhouette_gap=3,
                             gap_max_blob=500, gap_max_dist=10, gap_lum_thresh=140,
                             bg_tol=16):
    """Run the occlusion z-order, write synthetic <tag>_depth.png + info.json,
    then let the repo's further_extr() export the PSD. Returns the layer order."""
    d = layer_dir
    order, rels = solve(d, workers=workers, min_margin=min_margin, min_cov=min_cov, verbose=verbose)
    n = max(len(order), 1)

    # persist the solved order so downstream consumers (ComfyUI nodes, re-exports)
    # can composite or rebuild the PSD without re-running the solver
    try:
        with open(osp.join(layer_dir, "zorder.json"), "w", encoding="utf-8") as fh:
            json.dump({"order_bottom_first": order,
                       "relations": rels}, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        if verbose:
            print(f"[zorder] could not write zorder.json: {exc}", flush=True)

    if patch_layers:
        from patch_layers import patch
        print("[patch] repairing layer content with the source image", flush=True)
        patch(layer_dir, order, dilate=patch_dilate, color_thresh=patch_color_thresh,
              silhouette_gap=silhouette_gap, gap_max_blob=gap_max_blob,
              gap_max_dist=gap_max_dist, gap_lum_thresh=gap_lum_thresh,
              bg_tol=bg_tol, verbose=verbose)

    # bottom layer -> largest depth value (dump_parts_psd sorts descending)
    h = w = None
    parts = {}
    for idx, tag in enumerate(order):
        score = 1.0 - 0.9 * idx / max(n - 1, 1)          # 1.0 (bottom) .. 0.1 (top)
        img = Image.open(osp.join(d, tag + ".png")).convert("RGBA")
        if h is None:
            w, h = img.size
        depth = np.full((h, w), int(round(score * 255)), dtype=np.uint8)
        Image.fromarray(depth).save(osp.join(d, tag + "_depth.png"))
        parts[tag] = {}
        print(f"  {tag:<12} depth_median={score:.3f}")

    with open(osp.join(d, "info.json"), "w", encoding="utf-8") as fh:
        json.dump({"parts": parts, "frame_size": [h, w]}, fh, indent=2)

    if save_psd:
        iu.further_extr(d, rotate=False, save_to_psd=True, tblr_split=False)
    if verbose:
        print(f"[done] z-order solved for {len(order)} layers (no depth model used)", flush=True)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_margin", type=float, default=0.08)
    ap.add_argument("--min_cov", type=float, default=0.15)
    ap.add_argument("--save_psd", action="store_true")
    ap.add_argument("--patch", action="store_true", help="repair layer content using the source image")
    ap.add_argument("--patch_dilate", type=int, default=6)
    ap.add_argument("--patch_color_thresh", type=int, default=45)
    ap.add_argument("--silhouette_gap", type=int, default=3,
                    help="px around the outline that are restored from the source image")
    args = ap.parse_args()
    write_algo_depth_and_psd(args.layer_dir, workers=args.workers, min_margin=args.min_margin,
                             min_cov=args.min_cov, save_psd=args.save_psd,
                             patch_layers=args.patch, patch_dilate=args.patch_dilate,
                             patch_color_thresh=args.patch_color_thresh,
                             silhouette_gap=args.silhouette_gap)


if __name__ == "__main__":
    main()
