# ============================================================
# 图片质量检查：正式分析之前，先检查街景图是否合格
#
# 检查四项：
#   1. 天空占比 —— 太少说明画面可能混进了界面，或视角不对
#   2. 异常类别 —— bus/train/truck 占比过高，通常是误判
#   3. 亮度 —— 太暗的图分割会失准
#   4. 宽高比 —— 太宽的图会影响分割精度
#
# 用法：把图片放进 images/ 文件夹，然后运行本脚本
# ============================================================

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from glob import glob

import numpy as np
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

CITYSCAPES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# 判定阈值（可按需要调整）
SKY_MIN = 0.05         # 天空至少占 5%
ODD_CLASS_MAX = 0.10   # bus + train + truck 合计不该超过 10%
BRIGHT_MIN = 60        # 平均亮度至少 60
ASPECT_MAX = 2.0       # 宽高比不超过 2.0

print("正在加载模型……")
model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

files = sorted(glob("images/*.jpg") + glob("images/*.jpeg") + glob("images/*.png"))
if not files:
    print("images 文件夹里没有图片。")
    raise SystemExit(1)

print(f"\n共 {len(files)} 张图片，开始检查……\n")
print(f"{'文件':<16}{'尺寸':<13}{'亮度':<7}{'sky':<8}{'road':<8}{'bldg':<8}{'判定'}")
print("-" * 82)

problems = []
for f in files:
    img = Image.open(f).convert("RGB")
    w, h = img.size
    bright = float(np.array(img.convert("L")).mean())

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    logits = torch.nn.functional.interpolate(
        out.logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    pred = logits.argmax(dim=1)[0].numpy()
    total = pred.size
    props = {name: (pred == i).sum() / total for i, name in enumerate(CITYSCAPES)}

    # 逐项判定
    # 注意：天空少有两种可能——(1) 真问题（画面混进界面）
    #                        (2) 场景特征（林荫道、骑楼，树冠/建筑挡住了天空）
    # 如果植被很多，说明大概率是林荫道，属于正常场景，不算问题
    issues = []
    if props["sky"] < SKY_MIN and props["vegetation"] < 0.15:
        issues.append("天空过少")
    if props["bus"] + props["train"] + props["truck"] > ODD_CLASS_MAX:
        issues.append("异常类别")
    if bright < BRIGHT_MIN:
        issues.append("过暗")
    if w / h > ASPECT_MAX:
        issues.append("过宽")

    verdict = "OK" if not issues else " ".join(issues)
    if issues:
        problems.append((os.path.basename(f), verdict))

    print(
        f"{os.path.basename(f):<16}{str(w) + 'x' + str(h):<13}{bright:<7.0f}"
        f"{props['sky']:<8.1%}{props['road']:<8.1%}{props['building']:<8.1%}{verdict}"
    )

print()
if problems:
    print(f"有 {len(problems)} 张需要留意：")
    for name, why in problems:
        print(f"   {name}：{why}")
else:
    print("全部合格")
