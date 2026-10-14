"""see-through + Hyper-SD 4-step LoRA - one-shot entry point.

Verified config (no repo source changes required):
    LoRA        : Hyper-SDXL-4steps-lora.safetensors
    CFG         : 0
    scheduler   : DDIMScheduler(timestep_spacing='trailing')
    steps       : 4

Examples
--------
# layer decomposition only (fast smoke test)
python tools/fast_decomp/run_fast_decomp.py --srcp assets/test_image.png --skip_depth

# full pipeline -> layered PSD
python tools/fast_decomp/run_fast_decomp.py --srcp path/to/image.png --save_to_psd
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

import argparse, json, os, os.path as osp, sys, time

os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from diffusers import DDIMScheduler
from diffusers.loaders import peft as _peft
from utils import inference_utils as iu
from utils.torch_utils import seed_everything
from utils.io_utils import find_all_imgs
import stages as st

HYPER_SD_REPO = "ByteDance/Hyper-SD"
LORA_FILES = {
    "4steps": "Hyper-SDXL-4steps-lora.safetensors",
    "8steps_cfg": "Hyper-SDXL-8steps-CFG-lora.safetensors",
}


def lora_path(name):
    """Resolve a Hyper-SD LoRA from the local HF cache, downloading on first use."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HYPER_SD_REPO, LORA_FILES[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcp", default="assets/test_image.png")
    ap.add_argument("--save_dir", default=r"workspace\lora4")
    ap.add_argument("--lora", default="4steps", choices=list(LORA_FILES))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=0.0,
                    help="0 = CFG off (required by the 4-step LoRA)")
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--resolution_depth", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_to_psd", action="store_true")
    ap.add_argument("--skip_depth", action="store_true")
    ap.add_argument("--head_detail", choices=["on", "off"], default="off",
                    help="off = skip the head detail stage (tk_seethrough enable_head_detail=False)")
    ap.add_argument("--clean", action="store_true", help="remove LoRA residue on empty layers")
    ap.add_argument("--clean_preset", default="soft", choices=["soft", "mid"])
    ap.add_argument("--clean_dilate", type=int, default=40)
    ap.add_argument("--zorder", choices=["algo", "depth"], default="algo",
                    help="algo: occlusion-consistency z-order (no GPU, seconds); "
                         "depth: the original Marigold depth pass")
    ap.add_argument("--zorder_workers", type=int, default=8)
    ap.add_argument("--patch_layers", dest="patch_layers", action="store_true", default=True,
                    help="repair layer content with the source image after z-ordering (default on)")
    ap.add_argument("--no_patch", dest="patch_layers", action="store_false")
    ap.add_argument("--patch_dilate", type=int, default=6)
    ap.add_argument("--patch_color_thresh", type=int, default=45)
    ap.add_argument("--silhouette_gap", type=int, default=3,
                    help="px around the outline restored from the source image (kills the white fringe)")
    ap.add_argument("--group_offload", action="store_true")
    ap.add_argument("--repo_id_layerdiff", default="layerdifforg/seethroughv0.0.2_layerdiff3d")
    ap.add_argument("--repo_id_depth", default="24yearsold/seethroughv0.0.1_marigold")
    args = ap.parse_args()

    srcs = find_all_imgs(args.srcp, abs_path=True) if osp.isdir(args.srcp) else [args.srcp]
    for src in srcs:
        seed_everything(args.seed)
        t0 = time.time()
        iu.apply_layerdiff(src, args.repo_id_layerdiff, save_dir=osp.join(args.save_dir, "_warm"),
                           seed=args.seed, resolution=args.resolution,
                           disable_progressbar=True, num_inference_steps=1,
                           group_offload=args.group_offload)
        pipe = iu.layerdiff_pipeline
        build_s = time.time() - t0

        # --- LoRA + official 4-step sampling config -------------------------
        _peft._SET_ADAPTER_SCALE_FN_MAPPING.setdefault(
            type(pipe.unet).__name__, _peft._SET_ADAPTER_SCALE_FN_MAPPING["UNet2DConditionModel"])
        te1, te2 = pipe.text_encoder, pipe.text_encoder_2
        lora_file = lora_path(args.lora)
        pipe.load_lora_weights(lora_file, adapter_name="hyper_sd")
        pipe.text_encoder, pipe.text_encoder_2 = te1, te2      # fail loudly, not silently
        pipe.unet.to(dtype=torch.bfloat16)
        pipe.unet.set_adapters(["hyper_sd"], weights=[1.0])
        pipe.unet.fuse_lora(lora_scale=1.0, adapter_names=["hyper_sd"])
        pipe.unet.to(dtype=torch.bfloat16)

        known = set(DDIMScheduler().config.keys())
        cfg = {k: v for k, v in dict(pipe.scheduler.config).items() if k in known}
        cfg["timestep_spacing"] = "trailing"
        pipe.scheduler = DDIMScheduler(**cfg)

        cls, orig = type(pipe), type(pipe).__call__
        def patched(self, *a, __orig=orig, **kw):
            kw["guidance_scale"] = args.cfg
            return __orig(self, *a, **kw)
        cls.__call__ = patched
        print(f"[setup] lora={osp.basename(lora_file)} steps={args.steps} "
              f"cfg={args.cfg} scheduler=DDIM(trailing) build={build_s:.1f}s", flush=True)

        # --- layer decomposition -------------------------------------------
        # head_detail=off skips the 2nd (head) pass entirely: those 11 fine layers
        # are the ones the 4-step LoRA turns into residue.
        t1 = time.time()
        if args.head_detail == "off":
            st.run_body_stage(pipe, src, args.save_dir, resolution=args.resolution,
                              steps=args.steps, seed=args.seed, disable_progressbar=True)
        else:
            iu.apply_layerdiff(src, args.repo_id_layerdiff, save_dir=args.save_dir, seed=args.seed,
                               resolution=args.resolution, disable_progressbar=True,
                               num_inference_steps=args.steps, group_offload=args.group_offload)
        layerdiff_s = time.time() - t1

        # --- residue cleanup (before depth, so masks/PSD see clean layers) ---
        name = osp.basename(osp.splitext(src)[0])
        if args.clean:
            import subprocess
            cmd = [sys.executable, osp.join(HERE, "clean_residue.py"),
                   "--src_dir", osp.join(args.save_dir, name),
                   "--out_dir", osp.join(args.save_dir, name),
                   "--preset", args.clean_preset, "--dilate", str(args.clean_dilate)]
            subprocess.run(cmd, check=True)
            print(f"[clean] residue removed (preset={args.clean_preset}, dilate={args.clean_dilate})", flush=True)

        # --- layer ordering + PSD -------------------------------------------
        depth_s = 0.0
        if not args.skip_depth:
            t2 = time.time()
            if args.zorder == "algo":
                from build_psd import write_algo_depth_and_psd
                order = write_algo_depth_and_psd(osp.join(args.save_dir, name),
                                                 workers=args.zorder_workers,
                                                 save_psd=args.save_to_psd,
                                                 patch_layers=args.patch_layers,
                                                 patch_dilate=args.patch_dilate,
                                                 patch_color_thresh=args.patch_color_thresh,
                                                 silhouette_gap=args.silhouette_gap)
                print(f"[zorder] {len(order)} layers ordered by occlusion (no depth model)", flush=True)
            else:
                iu.apply_marigold(src, args.repo_id_depth, save_dir=args.save_dir, seed=args.seed,
                                  disable_progressbar=True, resolution=args.resolution_depth,
                                  num_inference_steps=-1, group_offload=args.group_offload)
                iu.further_extr(osp.join(args.save_dir, name), rotate=False,
                                save_to_psd=args.save_to_psd, tblr_split=False)
            depth_s = time.time() - t2

        out = osp.join(args.save_dir, osp.basename(osp.splitext(src)[0]))
        stats = dict(src=src, steps=args.steps, cfg=args.cfg, lora=args.lora,
                     build_s=round(build_s, 1), layerdiff_s=round(layerdiff_s, 1),
                     depth_s=round(depth_s, 1), out=out)
        print(f"[done] layerdiff={layerdiff_s:.1f}s depth={depth_s:.1f}s -> {out}", flush=True)
        with open(osp.join(args.save_dir, "run_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
