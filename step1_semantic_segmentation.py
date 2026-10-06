# 复现论文第一步：用预训练语义分割模型，把一张街景图分成 19 类
# 对应论文《Beyond single snapshots》的 3.1 节

import os

# 用国内镜像下载模型，避免卡住（如果你的网络能直连 Hugging Face，可以删掉这行）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，避免中文在图上显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Cityscapes 的 19 个类别，按编号顺序排列
CITYSCAPES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# 1. 加载预训练模型（在 Cityscapes 街景数据集上训练好的 SegFormer）
model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.eval()

# 2. 读一张街景图（自动取 images 文件夹里的第一张）
image_files = sorted(
    f for f in os.listdir("images")
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)
if not image_files:
    print("【提示】images 文件夹里没有图片，请先放入街景图。")
    raise SystemExit(1)
image_path = os.path.join("images", image_files[0])
print(f"正在处理：{image_path}")
image = Image.open(image_path).convert("RGB")

# 3. 预处理 + 推理
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits  # 形状 (1, 19, H/4, W/4)

# 4. 把结果放大回原图尺寸，每个像素取概率最大的那个类别
logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
pred = logits.argmax(dim=1)[0].numpy()  # 每个像素的类别编号

# 5. 统计每类像素占比——这就是后面算六个指标的原材料
total = pred.size
unique, counts = np.unique(pred, return_counts=True)
print("各类像素占比：")
for cls_id, cnt in zip(unique, counts):
    print(f"  {CITYSCAPES[cls_id]:15s} {cnt / total * 100:5.2f}%")

# 6. 可视化对比：原图 vs 分割结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("原图")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pred)
plt.title("语义分割结果（颜色 = 类别）")
plt.axis("off")
plt.tight_layout()

# 保存成图片文件，方便查看和放进报告
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/fig1_segmentation.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"结果图已保存到：{output_path}")

plt.show()
