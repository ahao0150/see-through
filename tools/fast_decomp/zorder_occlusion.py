r"""Depth-free layer ordering by occlusion consistency (v2).

Physical basis: for two overlapping layers, the *upper* one still holds the
original image pixels in the overlap (it was never hidden), while the *lower*
one was inpainted by the diffusion model and therefore deviates from the source.
So inside the overlap, whichever layer matches the source image better is on top.

v2 changes
  * coverage = |intersection| / min(|mask_i|, |mask_j|) - filters accidental
    neighbour overlap (a pair must actually cover each other to count)
  * only relations with |margin| >= --min_margin are kept as graph edges
  * ordering comes from a topological sort (edges added strongest-first, cycles
    dropped) instead of weighted vote summing
  * all pairs evaluated concurrently (numpy releases the GIL)

  python zorder_occlusion2.py --layer_dir <dir> --workers 8
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

import argparse, itertools, json, os, os.path as osp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image


# Anatomically hard constraints: (upper, lower). These are physical facts for an
# upright character, so they are forced regardless of what the pixel evidence says.
PRIOR_RULES = [
    ("front hair", "head"),
    ("head", "neck"),
    ("topwear", "neck"),
    ("topwear", "bottomwear"),
    ("bottomwear", "legwear"),
    ("footwear", "legwear"),        # a shoe is always in front of the foot/leg
    ("handwear", "topwear"),
    ("handwear", "bottomwear"),
]


def compare_pair(job):
    ti, ri, mi, tj, rj, mj, ro, min_margin, min_cov, sim_hi = job
    inter = mi & mj
    n = int(inter.sum())
    if n == 0:
        return None
    ai, aj = int(mi.sum()), int(mj.sum())
    cover = n / max(min(ai, aj), 1)
    if cover < min_cov:
        return None
    o = ro[inter].astype(np.float32)
    sim_i = 1.0 - float(np.abs(ri[inter].astype(np.float32) - o).mean()) / 255.0
    sim_j = 1.0 - float(np.abs(rj[inter].astype(np.float32) - o).mean()) / 255.0
    margin = sim_i - sim_j
    hi_i, hi_j = sim_i >= sim_hi, sim_j >= sim_hi
    confirmed = hi_i != hi_j                     # one side matches the source almost exactly
    if not confirmed and abs(margin) < min_margin:
        return None
    return dict(i=ti, j=tj, cover=round(cover, 3), sim_i=round(sim_i, 4), sim_j=round(sim_j, 4),
                margin=round(margin, 4), confirmed=bool(confirmed),
                on_top=ti if margin > 0 else tj)


# Semantic fallback rank (bottom -> top) used only to break ties between layers
# that have no direct occlusion evidence; it never overrides evidence or priors.
DEFAULT_RANK = {
    "back hair": 0, "wings": 1, "tail": 1, "objects": 2,
    "legwear": 3, "footwear": 4, "bottomwear": 5, "neck": 6, "topwear": 7,
    "handwear": 8, "ears": 9, "earwear": 10, "head": 11, "face": 12,
    "eyewear": 13, "eyebrow": 14, "eyelash": 14, "eyewhite": 14, "irides": 14,
    "nose": 14, "mouth": 14, "headwear": 15, "front hair": 16,
}


def topo_order(tags, edges, priors=None):
    """Kahn sort. Prior (anatomical) edges are forced; evidence edges are added
    strongest-first and dropped when they would create a cycle."""
    accepted = []
    if priors:
        ts = set(tags)
        for up, low in priors:
            if up in ts and low in ts:
                accepted.append(dict(i=low, j=up, on_top=up, margin=1.0, confirmed=True,
                                     cover=1.0, sim_i=1.0, sim_j=0.0, prior=True))
    def has_cycle(extra):
        adj = {t: [] for t in tags}
        for e in accepted + [extra]:
            adj[e["on_top"]].append(e["i"] if e["on_top"] == e["j"] else e["j"])
        state = {}
        def dfs(u):
            state[u] = 1
            for v in adj[u]:
                if state.get(v) == 1:
                    return True
                if state.get(v, 0) == 0 and dfs(v):
                    return True
            state[u] = 2
            return False
        return any(state.get(t, 0) == 0 and dfs(t) for t in tags)

    for e in edges:
        if not has_cycle(e):
            accepted.append(e)

    indeg = {t: 0 for t in tags}
    adj = {t: [] for t in tags}
    for e in accepted:
        lose = e["i"] if e["on_top"] == e["j"] else e["j"]
        adj[lose].append(e["on_top"])          # loses -> on_top
        indeg[e["on_top"]] += 1
    import heapq
    heap = [(DEFAULT_RANK.get(t, 8), t) for t in tags if indeg[t] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (DEFAULT_RANK.get(v, 8), v))
    return out, accepted


def solve(layer_dir, orig=None, workers=8, min_margin=0.08, min_cov=0.15, sim_hi=0.90,
          priors=None, skip="src_img,src_head,reconstruction", verbose=True):
    """Return (order_bottom_first, relations) for a directory of layer PNGs."""
    orig = orig or osp.join(layer_dir, "src_img.png")
    ro = np.asarray(Image.open(orig).convert("RGBA"))[..., :3]
    skip = set(skip.split(",") if isinstance(skip, str) else skip)

    masks, rgbs = {}, {}
    for f in sorted(os.listdir(layer_dir)):
        if not f.endswith(".png") or f[:-4] in skip or f.endswith("_depth.png"):
            continue
        a = np.asarray(Image.open(osp.join(layer_dir, f)).convert("RGBA"))
        m = a[..., 3] > 25
        if m.sum() < 100:
            continue
        masks[f[:-4]], rgbs[f[:-4]] = m, a[..., :3]
    tags = list(masks)
    if verbose:
        print(f"[zorder] layers with content: {len(tags)} -> {tags}", flush=True)

    jobs = [(i, rgbs[i], masks[i], j, rgbs[j], masks[j], ro, min_margin, min_cov, sim_hi)
            for i, j in itertools.combinations(tags, 2)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rels = [r for r in ex.map(compare_pair, jobs) if r]
    rels.sort(key=lambda r: (not r["confirmed"], -abs(r["margin"])))
    order, accepted = topo_order(tags, rels, priors if priors is not None else PRIOR_RULES)
    if verbose:
        n_conf = sum(1 for r in rels if r["confirmed"])
        print(f"[zorder] sim_hi={sim_hi} confirmed pairs={n_conf}/{len(rels)}; "
              f"edges used={len(accepted)} (bottom->top): {order}", flush=True)
    return order, rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_dir", required=True)
    ap.add_argument("--orig", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_margin", type=float, default=0.08)
    ap.add_argument("--min_cov", type=float, default=0.15)
    ap.add_argument("--sim_hi", type=float, default=0.90,
                    help="similarity at which a layer is confirmed as the top one")
    ap.add_argument("--skip", default="src_img,src_head,reconstruction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    orig = args.orig or osp.join(args.layer_dir, "src_img.png")
    ro = np.asarray(Image.open(orig).convert("RGBA"))[..., :3]
    skip = set(args.skip.split(","))

    masks, rgbs = {}, {}
    for f in sorted(os.listdir(args.layer_dir)):
        if not f.endswith(".png") or f[:-4] in skip or f.endswith("_depth.png"):
            continue
        a = np.asarray(Image.open(osp.join(args.layer_dir, f)).convert("RGBA"))
        m = a[..., 3] > 25
        if m.sum() < 100:
            continue
        masks[f[:-4]], rgbs[f[:-4]] = m, a[..., :3]
    tags = list(masks)
    print(f"layers with content: {len(tags)} -> {tags}")

    jobs = [(i, rgbs[i], masks[i], j, rgbs[j], masks[j], ro, args.min_margin, args.min_cov, args.sim_hi)
            for i, j in itertools.combinations(tags, 2)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rels = [r for r in ex.map(compare_pair, jobs) if r]
    rels.sort(key=lambda r: -abs(r["margin"]))

    print(f"\n{'pair':<26}{'cover':>7}{'simA':>7}{'simB':>7}{'margin':>8}  on_top")
    for r in rels:
        print(f"{r['i']+' vs '+r['j']:<26}{r['cover']:>7.2f}{r['sim_i']:>7.3f}{r['sim_j']:>7.3f}"
              f"{r['margin']:>8.3f}  {r['on_top']}")

    order_bottom_first, accepted = topo_order(tags, rels, PRIOR_RULES)
    print(f"\nkept {len(accepted)}/{len(rels)} relations after cycle removal")
    print("z-order (bottom -> top):")
    for k, t in enumerate(order_bottom_first):
        print(f"  {k:2d}  {t}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dict(order_bottom_first=order_bottom_first, relations=rels,
                           used_relations=accepted), fh, indent=2)
        print("json ->", args.out)


if __name__ == "__main__":
    main()
