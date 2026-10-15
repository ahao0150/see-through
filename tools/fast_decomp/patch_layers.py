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



def detect_backdrop_color(src_rgba, quant=8):
    """Estimate the backdrop colour as the dominant colour of the border ring.

    Works for any flat backdrop (white, green screen, ...). Returns an int16 RGB
    triple, or None when there is nothing to sample.
    """
    a = src_rgba[..., 3]
    rgb = src_rgba[..., :3].astype(np.int16)
    h, w = a.shape
    k = max(3, min(h, w) // 60)
    border = np.zeros(a.shape, bool)
    border[:k, :] = border[-k:, :] = True
    border[:, :k] = border[:, -k:] = True
    px = rgb[border & (a > 0)]
    if len(px) == 0:
        return None
    q = (px // quant) * quant
    vals, counts = np.unique(q, axis=0, return_counts=True)
    return vals[int(np.argmax(counts))].astype(np.int16)


def backdrop_like(src_rgba, bg_color, tol=16):
    """Pixels whose colour matches the backdrop within `tol` (per channel).

    Black and white backdrops are the supported cases; the tolerance is applied to
    the grey level, so a white backdrop test is "all channels >= 255-tol" and a
    black one is "all channels <= tol".
    """
    if bg_color is None:
        return np.zeros(src_rgba.shape[:2], bool)
    rgb = src_rgba[..., :3].astype(np.int16)
    bg = np.asarray(bg_color, np.int16)
    return (np.abs(rgb - bg).max(-1) <= tol) & (src_rgba[..., 3] > 0)


def _touches_border(m):
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def character_mask(src_rgba, bg_tol=16, min_bg_ratio=0.05, silent=False):
    """Mask of the character itself.

    The backdrop is the largest connected region matching the detected backdrop
    colour, provided it covers a decent share of the canvas and either touches the
    canvas border (no padding) or leans on the transparent padding. Everything else
    counts as character. Works with white, green-screen or any other flat backdrop.
    """
    from scipy import ndimage
    a = src_rgba[..., 3]
    person = a > 0
    bg_color = detect_backdrop_color(src_rgba)
    bg_like = backdrop_like(src_rgba, bg_color, bg_tol)
    if not silent:
        print(f"[backdrop] detected colour RGB{tuple(int(v) for v in bg_color)} "
              f"-> {int(bg_like.sum())} px match (tol={bg_tol})", flush=True)
    lab, n = ndimage.label(bg_like)
    if n:
        sizes = ndimage.sum(bg_like, lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] >= min_bg_ratio * bg_like.size:
            blob = lab == k
            leans_out = _touches_border(blob) or bool(
                (ndimage.binary_dilation(blob, iterations=2) & (a == 0)).any())
            if leans_out:
                person = person & ~blob
    return person


def complete_silhouette(layer_dir, order_bottom_first, max_gap=3, bg_tol=16, verbose=True):
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
    person = character_mask(src, bg_tol=bg_tol, silent=True)

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
    pockets = backdrop_gap_mask(src, bg_tol=bg_tol)
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


def backdrop_gap_mask(src_rgba, bg_tol=16, max_blob=500, max_dist=10, verbose=False):
    """Pure-white pockets that belong to the backdrop but are enclosed by the art.

    The artwork's backdrop is the largest white connected region. Any *other* white
    blob that is small (<= max_blob px) and sits close to that backdrop (<= max_dist
    px) is backdrop showing through a gap - e.g. between hair strands. White areas
    that are genuinely character content (a big white shoe) are larger or far from
    the backdrop and are kept.
    """
    from scipy import ndimage
    bg_color = detect_backdrop_color(src_rgba)
    white = backdrop_like(src_rgba, bg_color, bg_tol)
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
                         bg_tol=16, lum_thresh=140, verbose=True):
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
    bg_color = detect_backdrop_color(src)
    cands = backdrop_gap_mask(src, bg_tol=bg_tol, max_blob=max_blob, max_dist=max_dist)

    # Judge every white blob on its own (labelling the merged candidate mask would
    # fuse an eye highlight with a nearby hair gap and clear both).
    white_all = backdrop_like(src, bg_color, bg_tol)
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


def backdrop_region(src_rgba, bg_color, tol=16):
    """The backdrop *region*: colour-matching pixels connected to the canvas border.

    Colour matching alone cannot tell a white backdrop from the character's white
    clothing, so the region is restricted to the component that reaches the border
    or leans on the transparent padding. Enclosed white content (socks, shirts,
    highlights) is left alone.
    """
    from scipy import ndimage as _nd
    if bg_color is None:
        return np.zeros(src_rgba.shape[:2], bool)
    a = src_rgba[..., 3]
    near = backdrop_like(src_rgba, bg_color, tol)
    lab, n = _nd.label(near)
    if n == 0:
        return near
    keep = set()
    for idx in np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])):
        if idx != 0:
            keep.add(int(idx))
    if not keep:
        return near
    region = np.isin(lab, list(keep))
    # transparent padding (from centre_square_pad_resize) also counts as backdrop
    region |= (a == 0)
    return region


def backdrop_kind(bg_color):
    """Classify the detected backdrop. Only white and black are colour-keyed.

    A tinted backdrop (pink, green screen, ...) is too close to pale character
    colours for any per-channel tolerance: the sample pink background is within
    16/255 of the character's skin, so colour keying there eats the skin. For
    those we refuse to key and keep the layers intact.
    """
    b = np.asarray(bg_color, np.float32)
    # near-white additionally demands low saturation: the sample pink backdrop
    # (248, 216, 216) has min=216 and would otherwise pass a brightness-only test,
    # and pale skin sits within its tolerance.
    if b.min() >= 200.0 and (b.max() - b.min()) <= 24.0:
        return "white"
    if b.max() <= 56.0:
        return "black"
    return None


def drop_backdrop_pixels(layer_dir, order_bottom_first, bg_tol=16, verbose=True):
    """Make layer pixels that *are* the backdrop colour transparent.

    Deliberately conservative: no colour reconstruction, no alpha estimation, so
    outlines are never touched. Only pixels whose colour matches the detected
    backdrop (within bg_tol) lose their alpha - that is what removes the white /
    black fringe that anti-aliasing blends into the layer border.
    """
    src = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    bg = detect_backdrop_color(src)
    region = backdrop_region(src, bg, bg_tol)
    kind = backdrop_kind(bg)
    if kind is None:
        if verbose:
            print(f"[drop-backdrop] backdrop RGB{tuple(int(v) for v in bg)} is neither white "
                  f"nor black - colour keying skipped (character content would be eaten)",
                  flush=True)
        return 0
    removed = 0
    per = []
    for tag in order_bottom_first:
        p = osp.join(layer_dir, tag + ".png")
        if not osp.exists(p):
            continue
        img = np.asarray(Image.open(p).convert("RGBA")).copy()
        m = img[..., 3] > 0
        hit = m & region
        n = int(hit.sum())
        if n:
            img[hit, 3] = 0
            Image.fromarray(img).save(p)
            removed += n
            per.append((tag, n))
    if verbose:
        print(f"[drop-backdrop] removed {removed} backdrop-coloured pixels from layers "
              f"{per}", flush=True)
    return removed


def patch(layer_dir, order_bottom_first, out_dir=None, dilate=6, color_thresh=45,
          silhouette_gap=3, gap_max_blob=500, gap_max_dist=10, gap_lum_thresh=140,
          bg_tol=16, verbose=True):
    out_dir = out_dir or layer_dir
    os.makedirs(out_dir, exist_ok=True)
    orig = np.asarray(Image.open(osp.join(layer_dir, "src_img.png")).convert("RGBA"))
    person = character_mask(orig, bg_tol=bg_tol, silent=True)

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

    # Order matters: the silhouette is completed LAST so the artwork's own
    # anti-aliased rim (near-white pixels next to a white backdrop, which the colour
    # key would otherwise drop) is put back from the source image.
    remove_backdrop_gaps(layer_dir, order_bottom_first, max_blob=gap_max_blob,
                         max_dist=gap_max_dist, bg_tol=bg_tol, lum_thresh=gap_lum_thresh,
                         verbose=verbose)
    drop_backdrop_pixels(layer_dir, order_bottom_first, bg_tol=bg_tol, verbose=verbose)
    complete_silhouette(layer_dir, order_bottom_first, max_gap=silhouette_gap, bg_tol=bg_tol,
                        verbose=verbose)
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
    ap.add_argument("--bg_tol", type=int, default=16,
                    help="per-channel tolerance when matching the backdrop (black/white)")
    ap.add_argument("--gap_lum_thresh", type=float, default=140,
                    help="pockets whose surroundings are brighter than this are protected (eyes)")
    args = ap.parse_args()
    order = json.load(open(args.order, encoding="utf-8"))["order_bottom_first"]
    print(f"z-order (bottom->top): {order}")
    patch(args.layer_dir, order, args.out_dir, dilate=args.dilate,
          color_thresh=args.color_thresh, silhouette_gap=args.silhouette_gap,
          gap_max_blob=args.gap_max_blob, gap_max_dist=args.gap_max_dist,
          gap_lum_thresh=args.gap_lum_thresh, bg_tol=args.bg_tol)


if __name__ == "__main__":
    main()
