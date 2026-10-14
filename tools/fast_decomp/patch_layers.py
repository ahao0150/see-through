r"""Patch layer content so the z-ordered stack reproduces the source image exactly.

Two operations, applied top layer -> bottom layer:

  1. colour-guided hole filling: inside a small dilation of a layer's mask, pixels
     that match the layer's dominant colour (and are still unclaimed) are adopted
     by that layer. Fixes e.g. a hand that was split off into the shorts layer.
  2. visible-region correction: whatever part of the layer is actually visible in
     the final stack (mask & remaining) is overwritten with the source pixels, so
     the recomposited result equals the original artwork pixel-for-pixel, while
     the hidden (inpainted) region stays as-is for animation use.

  python patch_layers.py --layer_dir <dir> --order <zorder.json> [--dilate 6]
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

import argparse, json, os, os.path as osp, sys
import numpy as np
from PIL import Image
from scipy import ndimage



def patch(layer_dir, order_bottom_first, out_dir=None, dilate=6, color_thresh=45, verbose=True):
    out_dir = out_dir or layer_dir
    os.makedirs(out_dir, exist_ok=True)
    orig = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    person = orig[..., 3] > 25

    imgs = {}
    for tag in order_bottom_first:
        p = osp.join(layer_dir, tag + ".png")
        if osp.exists(p):
            imgs[tag] = np.asarray(Image.open(p).convert("RGBA")).copy()

    remaining = person.copy()
    stats = {}
    for tag in reversed(order_bottom_first):            # top layer first
        img = imgs[tag]
        m = img[..., 3] > 25
        guard = remaining & ~m
        grown = int(guard.sum())

        # 1) colour-guided adoption of nearby unclaimed pixels of the same colour
        if m.any():
            region = ndimage.binary_dilation(m, iterations=dilate) & guard
            if region.any():
                main = np.median(img[m][:, :3].astype(np.int16), axis=0)
                dist = np.abs(orig[..., :3].astype(np.int16) - main).sum(-1)
                add = region & (dist < color_thresh)
                if add.any():
                    img[add] = orig[add]
                    m = m | add

        # 2) the visible part must equal the source image
        vis = m & remaining
        if vis.any():
            img[vis, :3] = orig[vis, :3]
            img[vis, 3] = 255
        remaining &= ~m

        Image.fromarray(img).save(osp.join(out_dir, tag + ".png"))
        v = int(vis.sum())
        stats[tag] = dict(pixels=int(m.sum()), visible=v, adopted=v and (int(m.sum()) - grown) or 0)
        if verbose:
            print(f"  {tag:<12} mask_px={int(m.sum()):>7} visible={v:>7}")

    left = int(remaining.sum())
    if verbose:
        print(f"[patch] unclaimed person pixels after patching: {left}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_dir", required=True)
    ap.add_argument("--order", required=True, help="zorder json with order_bottom_first")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--color_thresh", type=int, default=45)
    args = ap.parse_args()
    order = json.load(open(args.order, encoding="utf-8"))["order_bottom_first"]
    print(f"z-order (bottom->top): {order}")
    patch(args.layer_dir, order, args.out_dir, dilate=args.dilate,color_thresh=args.color_thresh)


if __name__ == "__main__":
    main()
