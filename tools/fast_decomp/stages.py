"""see-through stage control (mirrors tk_seethrough's enable_head_detail).

The v3 model runs two diffusion passes:
  body stage -> 13 layers (front/back hair, head, neck, neckwear, topwear,
                handwear, bottomwear, legwear, footwear, tail, wings, objects)
  head stage -> 11 fine layers (headwear, face, irides, eyebrow, eyewhite,
                eyelash, eyewear, ears, earwear, nose, mouth)

With enable_head_detail=False the second pass is skipped entirely: ~50% less
compute and the 11 noisiest (smallest) layers are never generated.
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

import os
import os.path as osp
import sys

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image

from utils.cv import center_square_pad_resize

BODY_TAGS = ["front hair", "back hair", "head", "neck", "neckwear", "topwear",
             "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects"]
HEAD_TAGS = ["headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash",
             "eyewear", "ears", "earwear", "nose", "mouth"]


def run_body_stage(pipe, srcp, save_dir, resolution=768, steps=4, seed=42,
                   disable_progressbar=True):
    """Body stage only - equivalent to tk_seethrough enable_head_detail=False."""
    tag_version = pipe.unet.get_tag_version()
    if tag_version != "v3":
        raise ValueError(f"body-only path implemented for v3 models, got {tag_version}")

    pipe.cache_tag_embeds()
    pipe.set_progress_bar_config(disable=disable_progressbar)

    saved = osp.join(save_dir, osp.splitext(osp.basename(srcp))[0])
    os.makedirs(saved, exist_ok=True)

    input_img = np.array(Image.open(srcp).convert("RGBA"))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    Image.fromarray(fullpage).save(osp.join(saved, "src_img.png"))

    rng = torch.Generator(device=pipe.unet.device).manual_seed(seed)
    out = pipe(strength=1.0, num_inference_steps=steps, batch_size=1, generator=rng,
               guidance_scale=1.0, prompt=BODY_TAGS, negative_prompt="",
               fullpage=fullpage, group_index=0)
    for rst, tag in zip(out.images, BODY_TAGS):
        Image.fromarray(rst).save(osp.join(saved, f"{tag}.png"))
    return saved
