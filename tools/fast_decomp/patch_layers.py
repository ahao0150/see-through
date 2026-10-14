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



def character_mask(src_rgba, white_thresh=245, min_bg_ratio=0.05):
    """Mask of the character itself.

    Test images are often painted on an opaque white backdrop (every backdrop pixel
    has alpha=255), so the alpha channel only marks the padded area and cannot be
    trusted. The backdrop is therefore found as the largest pure-white connected
    region whose size is at least `min_bg_ratio` of the canvas and which touches the
    transparent padding (i.e. it sits outside the character). White areas *inside*
    the character (highlights, white shoes, 98 px in the sample) are far smaller and
    stay part of the character.
    """
    from scipy import ndimage
    a = src_rgba[..., 3]
    rgb = src_rgba[..., :3].astype(np.int16)
    white = (rgb >= white_thresh).all(-1)
    person = a > 0
    lab, n = ndimage.label(white)
    if n:
        sizes = ndimage.sum(white, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] >= min_bg_ratio * white.size:
            blob = lab == k
            outside = (a == 0) | np.zeros_like(a, bool)
            touches_outside = ndimage.binary_dilation(blob, iterations=2) & outside
            if touches_outside.any():
                person = person & ~blob
    return person


def complete_silhouette(layer_dir, order_bottom_first, max_gap=3, verbose=True):
    """Close the anti-aliased rim gap between the layer masks and the artwork.

    The decomposed layers' masks stop just inside the artwork's silhouette, so the
    artwork's own anti-aliased edge pixels (and any 1-2px gap along the outline)
    belong to no layer at all and show up as a pale fringe once composited.

    Fix: every painted-but-uncovered pixel within `max_gap` px of a layer is handed
    to the layer that is visible next to it, and its RGB **and alpha** are copied
    from the source image - so a pixel that is transparent in the source stays
    transparent, and one that is an outline stroke keeps the stroke.
    """
    from scipy import ndimage
    src = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    sa = src[..., 3]
    person = character_mask(src)

    tags = [t for t in order_bottom_first if osp.exists(osp.join(layer_dir, t + ".png"))]
    imgs = {t: np.asarray(Image.open(osp.join(layer_dir, t + ".png")).convert("RGBA")).copy()
            for t in tags}

    # visible owner of each covered pixel: top-most layer wins
    owner = np.full(sa.shape, -1, np.int32)
    for idx, tag in enumerate(reversed(tags)):          # top -> bottom
        m = imgs[tag][..., 3] > 0
        owner[m & (owner < 0)] = idx

    uncovered = owner < 0
    dist, inds = ndimage.distance_transform_edt(uncovered, return_indices=True)
    nearest = owner[inds[0], inds[1]]
    pockets = backdrop_gap_mask(src)
    gap = uncovered & (sa > 0) & person & ~pockets & (dist <= max_gap) & (nearest >= 0)
    if verbose:
        print(f"[silhouette] rim pixels restored: {int(gap.sum())} (within {max_gap}px)")

    for idx, tag in enumerate(reversed(tags)):
        add = gap & (nearest == idx)
        if add.any():
            imgs[tag][add] = src[add]                   # RGB + alpha straight from the art
            Image.fromarray(imgs[tag]).save(osp.join(layer_dir, tag + ".png"))

    # nothing may exist outside the character (keeps the PSD background transparent)
    for tag in tags:
        img = imgs[tag]
        bad = (img[..., 3] > 0) & ~person
        if bad.any():
            img[bad, 3] = 0
            Image.fromarray(img).save(osp.join(layer_dir, tag + ".png"))
    return int(gap.sum())


def backdrop_gap_mask(src_rgba, white_thresh=245, max_blob=500, max_dist=10,
                      gap_layers=None, src_path=None, verbose=False):
    """Pure-white pockets that belong to the backdrop but are enclosed by the art.

    The artwork's backdrop is the largest white connected region. Any *other* white
    blob that is small (<= max_blob px) and sits close to that backdrop (<= max_dist
    px) is backdrop showing through a gap - e.g. between hair strands. White areas
    that are genuinely character content (a big white shoe) are larger or far from
    the backdrop and are kept.
    """
    from scipy import ndimage
    a = src_rgba[..., 3]
    white = (src_rgba[..., :3] >= white_thresh).all(-1) & (a > 0)
    lab, n = ndimage.label(white)
    if n == 0:
        return np.zeros(white.shape, bool)
    sizes = ndimage.sum(white, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    dist = ndimage.distance_transform_edt(lab != main)
    gaps = np.zeros(white.shape, bool)
    for i in range(1, n + 1):
        if i == main:
            continue
        m = lab == i
        if sizes[i - 1] > max_blob or float(dist[m].min()) > max_dist:
            continue
        gaps |= m
    return gaps


def remove_backdrop_gaps(layer_dir, order_bottom_first, max_blob=500, max_dist=10,
                         white_thresh=245, lum_thresh=140, verbose=True):
    """Clear backdrop pockets from every layer so the PSD stays transparent there.

    Candidate pockets come from geometry (small white blob next to the backdrop).
    Each candidate is then judged on the *composited* stack: the ring of pixels
    around it must be dark (hair, mean luminance < lum_thresh) for the pocket to
    be backdrop showing through. Bright surroundings - face, eyes, glasses - are a
    safety zone and are left untouched, and every layer is cleaned at the same
    coordinates so no layer keeps the white.
    """
    from scipy import ndimage as _nd
    src = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    cands = backdrop_gap_mask(src, white_thresh=white_thresh, max_blob=max_blob, max_dist=max_dist)

    # Judge every white blob on its own (labelling the merged candidate mask would
    # fuse an eye highlight with a nearby hair gap and clear both).
    white_all = (src[..., :3] >= white_thresh).all(-1) & (src[..., 3] > 0)
    lab, n = _nd.label(white_all)
    sizes = _nd.sum(white_all, lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    dist_bg = _nd.distance_transform_edt(lab != main)
    gaps = np.zeros(white_all.shape, bool)
    protected = 0
    for i in range(1, n + 1):
        if i == main:
            continue
        m = lab == i
        if sizes[i - 1] > max_blob or float(dist_bg[m].min()) > max_dist:
            continue
        ring = _nd.binary_dilation(m, iterations=3) & ~white_all
        # judge on the *artwork*, immune to the patching done above
        if ring.any() and float(src[..., :3][ring].mean()) >= lum_thresh:
            protected += int(m.sum())          # bright surroundings -> safety zone
            continue
        gaps |= m
    if verbose:
        print(f"[backdrop-gaps] {int(gaps.sum())} pocket px to clear, "
              f"{protected} px protected by the bright-region safety zone", flush=True)
    removed = 0
    for tag in order_bottom_first:
        p = osp.join(layer_dir, tag + ".png")
        if not osp.exists(p):
            continue
        img = np.asarray(Image.open(p).convert("RGBA")).copy()
        cand = (img[..., 3] > 0) & gaps
        n = int(cand.sum())
        if n:
            img[cand, 3] = 0
            Image.fromarray(img).save(p)
            removed += n
            if verbose:
                print(f"  {tag:<12} cleared {n} backdrop pockets px")
    if verbose:
        print(f"[backdrop-gaps] cleared {removed} px from layers", flush=True)
    return removed


def patch(layer_dir, order_bottom_first, out_dir=None, dilate=6, color_thresh=45,
          silhouette_gap=3, gap_max_blob=500, gap_max_dist=10, gap_lum_thresh=140,
          verbose=True):
    out_dir = out_dir or layer_dir
    os.makedirs(out_dir, exist_ok=True)
    orig = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    person = character_mask(orig)

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

    complete_silhouette(layer_dir, order_bottom_first, max_gap=silhouette_gap, verbose=verbose)
    remove_backdrop_gaps(layer_dir, order_bottom_first, max_blob=gap_max_blob,
                         max_dist=gap_max_dist, lum_thresh=gap_lum_thresh, verbose=verbose)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_dir", required=True)
    ap.add_argument("--order", required=True, help="zorder json with order_bottom_first")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dilate", type=int, default=6)
    ap.add_argument("--color_thresh", type=int, default=45)
    ap.add_argument("--silhouette_gap", type=int, default=3)
    ap.add_argument("--gap_max_blob", type=int, default=500,
                    help="largest white pocket treated as backdrop (px)")
    ap.add_argument("--gap_max_dist", type=int, default=10,
                    help="max distance of a pocket to the backdrop (px); real hair gaps are "
                         "within a few px, eyes/glasses sit 25-35 px away (safety margin)")
    ap.add_argument("--gap_lum_thresh", type=float, default=140,
                    help="pockets whose surroundings are brighter than this are protected (eyes)")
    args = ap.parse_args()
    order = json.load(open(args.order, encoding="utf-8"))["order_bottom_first"]
    print(f"z-order (bottom->top): {order}")
    patch(args.layer_dir, order, args.out_dir, dilate=args.dilate,
          color_thresh=args.color_thresh, silhouette_gap=args.silhouette_gap,
          gap_max_blob=args.gap_max_blob, gap_max_dist=args.gap_max_dist,
          gap_lum_thresh=args.gap_lum_thresh)


if __name__ == "__main__":
    main()
