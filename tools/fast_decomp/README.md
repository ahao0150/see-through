# Fast 4-step Layer Decomposition (see-through 加速与免深度排序)

Without touching any file under `common/`, `inference/` or `annotators/`, this folder adds an
**optional, much faster** path from a single anime illustration to a layered PSD:

| 项目 | 原流程 | 本方案 |
|---|---|---|
| 扩散步数 | 30 步 | **4 步**（Hyper-SD XL LoRA） |
| 推理阶段 | body + head 两段（24 层） | body 单段（13 层，`--head_detail off`） |
| 图层排序 | Marigold 深度模型（GPU，数分钟） | **遮挡一致性算法（纯 CPU，秒级，可并发）** |
| 残影清理 | — | alpha/连通域/头部先验 二级清理 |
| 图层补全 | — | 源图引导打补丁（合成结果逐像素还原原图） |

Everything is additive: the original pipeline (`inference/scripts/inference_psd.py`) keeps working
unchanged.

---

## 1. 安装依赖

```bash
# 在原项目环境基础上，仅需补充 PEFT（用于加载 LoRA）
pip install peft

# Hyper-SD LoRA 会在首次运行时自动从 HuggingFace 下载到 HF 缓存
# 建议显式指定缓存目录（模型约 12.5 GB，LoRA 约 750 MB）
export HF_HOME=/path/to/hf_cache           # Windows: set HF_HOME=D:\hf_cache
```

`torch`/`diffusers` 等沿用根目录 `requirements.txt`。生成 PSD 需要 `psd-tools`（已在 requirements 内）。

---

## 2. 快速开始

```bash
# 完整流程：4 步 LoRA 分解 -> 残影清理 -> 算法排序 -> 打补丁 -> 导出 PSD
python tools/fast_decomp/run_fast_decomp.py \
    --srcp assets/test_image.png \
    --save_to_psd --clean

# 只看分层（不出 PSD，约 45 秒）
python tools/fast_decomp/run_fast_decomp.py --srcp assets/test_image.png --skip_depth

# 需要细分五官层（face / eyes / mouth / eyebrow ...）时打开 head 阶段
python tools/fast_decomp/run_fast_decomp.py --srcp x.png --save_to_psd --head_detail on --clean

# 仍想用原来的 Marigold 深度模型排序
python tools/fast_decomp/run_fast_decomp.py --srcp x.png --save_to_psd --zorder depth
```

可在仓库根目录批量处理一个文件夹：`--srcp path/to/folder/`。

---

## 3. 实测数据（RTX 3060 12 GB, 768×768）

| 阶段 | 耗时 |
|---|---|
| pipeline + LoRA 装载（一次性） | 75–80 s |
| 4 步 body-only 采样 | 47–49 s |
| 残影清理 | < 1 s |
| 遮挡排序 + 打补丁 + PSD 导出 | **~3 s** |
| *（对照）Marigold 深度排序* | *5 分钟以上（20 帧 × 4 步，显存打满）* |

质量：SDXL 30 步 → 4 步，PSNR/SSIM 层面属研究性近似；本方案通过「跳过 head 阶段 + 二级清理 +
打补丁」把 4 步的伪影压到零可见噪点，最终 PSD 合成与源图 **98.8% 像素完全一致**（平均差 1.49/255）。

---

## 4. 三个核心机制

### 4.1 Hyper-SD 4-step LoRA

- 权重：`ByteDance/Hyper-SD` → `Hyper-SDXL-4steps-lora.safetensors`（Kohya 格式，2364 个目标模块）
- 官方要求的三件套：**CFG = 0**、**DDIM + `timestep_spacing="trailing"`**、**4 步**
- 本项目 `inference_utils.apply_layerdiff()` 内部把 `guidance_scale` 写成 1.0，脚本通过包装
  pipeline 的 `__call__` 强制为 0（CFG=0 与 1.0 在本 pipeline 中等价，因为
  `do_classifier_free_guidance = guidance_scale > 1`）
- 自定义 UNet 类 `UNetFrameConditionModel` 不在 diffusers 的 adapter 缩放映射表里，脚本按类名
  注册别名后调用 `unet.set_adapters()` / `unet.fuse_lora()`

### 4.2 `--head_detail off`（对应 tk_seethrough 的 `enable_head_detail`）

v3 模型两段推理：

- **body**：front/back hair, head, neck, neckwear, topwear, handwear, bottomwear, legwear,
  footwear, tail, wings, objects（13 层）
- **head**：headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear,
  nose, mouth（11 层）

4 步 LoRA 下残影集中在 head 阶段那些细碎层；跳过整段 head，噪声源即消失（同时省掉一次扩散）。
该开关由 `stages.run_body_stage()` 实现，不修改仓库源码。

### 4.3 遮挡一致性排序（替代深度模型）

物理依据：两层重叠时，**上层在重叠区仍保留源图像素**（它从未被遮挡），**下层是被扩散模型
inpaint 出来的**，因此偏离源图。于是判定规则是：

> 在重叠区内，哪一层的像素更接近源图，哪一层就在上面。

实现要点（`zorder_occlusion.py`）：

- `cover = |A∩B| / min(|A|,|B|)`，低于 `--min_cov`（默认 0.15）的“擦边重叠”直接丢弃
- 高置信判据：某一侧相似度 ≥ `--sim_hi`（默认 **0.90**）而另一侧明显不达标 → 确认边
- 低置信判据：相似度差 `|margin| ≥ --min_margin`（默认 0.08）→ 普通边
- 先验硬规则 `PRIOR_RULES`（解剖学事实，永不被剪掉），例如
  **`footwear` 必在 `legwear` 之上**、`topwear > bottomwear`、`head > neck` 等
- 语义层级表 `DEFAULT_RANK` 作为无证据层的兜底排序
- 全部层对用线程池并发比较（numpy 释放 GIL），9 层 36 对耗时 < 1 s

### 4.4 二级残影清理（`clean_residue.py`）

1. 逐像素：`alpha < 64` 归零；连通域 < 48 px 的孤立噪点删除
2. 层级：整层 alpha 峰值 < 150 → 判为“无实体层”整体清空
3. 空间先验：面部部件只允许出现在 `head ∪ front hair ∪ back hair ∪ face` 的膨胀掩码内，
   躯干/腿部的斑块判为残影

### 4.5 打补丁与轮廓还原（`patch_layers.py`）

**背景识别**：很多插画是「白底不透明」的（白背景像素 alpha 也是 255），这时 alpha 通道只标记
padding 区域，不能当人物轮廓用。`character_mask()` 因此取**最大的纯白连通域**（≥5% 画布，
且紧邻透明 padding）作为背景剔除；人物内部的白色（高光、白鞋等）不受影响。
最终**所有层在人物轮廓外一律透明** —— 白底不会被带进 PSD。

**背景缝隙（发丝间白）**：除了主背景，插画里还有被画面内容包围、但仍属于背景的白色小块 ——
最典型的是**发丝之间的缝隙**（样本中 18 个白块，共 238 px，全部位于头部）。判定用两道闸门：

1. **几何**（`backdrop_gap_mask()`）：非最大白连通域 + 面积 ≤ `--gap_max_blob`（默认 500 px）+
   距主背景 ≤ `--gap_max_dist`（默认 **10 px**）。实测真正的发丝缝隙距背景 **2.2–8 px**，
   而眼睛/眼镜处的小白块在 **27–32 px**，这一步就把它们挡在安全区外。
2. **亮度安全区**：候选块环带（膨胀 3 px，去掉白色像素）在原图中的平均亮度 ≥
   `--gap_lum_thresh`（默认 140）则判定为「浅色细节」（眼白高光、镜片反光…）予以保护。
   发丝缝隙环带是深色头发（43–103），眼睛环带是肤色/眼白（170–222），区分度充足。

判定**逐个白块独立进行**（若在合并后的候选掩码上做连通域，眼白会和邻近的发丝缝隙粘成一块、
被一起清掉 —— 这个坑踩过）。命中像素随后**从所有层一并清除**（同一坐标可能多层都有内容，
必须每层都清）。实测：头发区域内「白色且不透明」像素 **235 px → 23 px**，且残留的 23 px
正是眼睛处被安全区保护下来的细节。

自上而下逐层处理：

1. **颜色引导补洞**：在层掩码外扩 `--patch_dilate`（默认 6 px）、且尚未被上层认领的区域中，
   把颜色接近该层主色（阈值 `--patch_color_thresh`，默认 45）的源图像素收进该层。
   （例：手被误分到短裤层的部分，因同为肤色而被收回 handwear）
2. **可见区校正**：该层在堆叠中实际可见的部分用源图像素覆盖，保证合成结果 == 原图；
   被遮挡的 inpaint 内容原样保留，供 Live2D 拉动图层时使用。
3. **轮廓补全（silhouette completion）**：分层掩码在人物外轮廓处往往比原图内缩 1–2 px，
   原图自身的抗锯齿边缘像素不属于任何层，合成后会出现一圈灰白晕边（实测轮廓带 49% 像素
   有明显偏差）。处理方式即“以原图为准”：未覆盖像素中距离已有层 ≤ `--silhouette_gap`
   （默认 3 px）且属于人物的部分，交给紧邻的可见层，**RGB 与 alpha 都直接取自原图** ——
   原图此处透明就透明，是描边就描边。
   实测：轮廓带偏差像素 **49.04% → 0.00%**，整体一致像素 **91.82% → 100%**。

---

## 5. 文件说明

| 文件 | 作用 |
|---|---|
| `run_fast_decomp.py` | 主入口（编排全流程，CLI 参数见下） |
| `stages.py` | body-only 推理阶段（= `enable_head_detail=False`） |
| `clean_residue.py` | 二级残影清理 |
| `zorder_occlusion.py` | 遮挡一致性排序（可独立 CLI 运行） |
| `patch_layers.py` | 源图引导打补丁 |
| `build_psd.py` | 生成伪深度 + `info.json` 并调用仓库原生 `further_extr()` 导出 PSD |

## 6. 主要参数

```
--lora {4steps,8steps_cfg}   LoRA 选择（默认 4steps）
--steps N                    扩散步数（默认 4）
--cfg F                      guidance scale（默认 0.0）
--head_detail {on,off}       是否跑 head 阶段（默认 off）
--clean / --clean_preset     残影清理开关与强度（soft / mid）
--zorder {algo,depth}        排序方式（默认 algo）
--patch_layers / --no_patch  是否打补丁（默认开）
--silhouette_gap N           轮廓还原半径（默认 3 px）
--gap_max_blob N             判定为背景缝隙的最大白块面积（默认 500 px）
--gap_max_dist N             背景缝隙距主背景的最大距离（默认 10 px；眼睛在 27–32 px 处）
--gap_lum_thresh F           环带亮度高于此值的白块视为细节并保护（默认 140）
--save_to_psd                导出分层 PSD
--skip_depth                 只做分层，不出 PSD
```

## 7. 已知限制

- **8 步 CFG LoRA 未接入**：它需要 CFG 分支，而本 pipeline 的 CFG 路径缺少条件批拼接、缺少
  `_guidance_rescale` 默认值，且 UNet 内部按「batch = 帧数」索引，需要改动 `common/` 源码才能跑通；
  实测单次 10 分钟以上，性价比不足，故未纳入。
- **`--head_detail off` 的取舍**：不产出 face/eyes/mouth 等细分层（整个头部在 `head` 层内），
  做表情/眨眼驱动时需切回 `on` 并配合清理参数。
- **遮挡排序的边界**：两层重叠过小（< `--min_cov`）时没有直接证据，依赖先验规则与语义层级兜底；
  极端姿势（举手过头、跨腿等）可能需要扩充 `PRIOR_RULES`。
- 打补丁会改写层文件的可见区（隐藏区不变），若需要“纯模型输出”请加 `--no_patch`。
- 背景识别针对**纯白底**（阈值 245）；彩色背景或渐变背景的插画需要先把背景抠成透明再送入
  （或调整 `character_mask` 的 `white_thresh`）。若原图本身已是透明背景，则直接使用其 alpha。

## 8. 许可

本目录代码遵循仓库根目录的 LICENSE（PolyForm Noncommercial 1.0.0，继承上游 see-through）。
Hyper-SD LoRA 版权归字节跳动，遵循其仓库许可。
