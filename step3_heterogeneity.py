# ============================================================
# 复现论文第三步：计算"视觉异质性"
# 对应论文《Beyond single snapshots》的 3.2 节
#
# 【核心思想（一句话）】
# 异质性 = 一张图周围"邻居图"在六个指标上的【方差】
#   方差大 → 周围变化剧烈 → 异质（有特色）
#   方差小 → 周围千篇一律 → 均质（单调）
#
# 【两个尺度（对应论文的 IH 和 CH）】
#   IH（即时异质性）：只看"眼前几步"，用"前后各 1 张"模拟
#   CH（背景异质性）：看"一整片街区"，用"前后各 3 张"模拟
#
# 【重要：这是简化版】
# 论文用的是"沿路网 50 米 / 500 米"来找邻居。
# 这里简化成"按文件名顺序，前后各 N 张"——因为你现在的图片没有坐标。
# 如果你以后有每张图的经纬度，可以把"按顺序找邻居"换成"按距离找邻居"。
#
# 【输入】images 文件夹里的一串街景图（至少 4–5 张，才能算异质性）
# 【输出】heterogeneity.csv（每张图的 IH、CH）+ figure_3.png
# ============================================================

import os

# 用国内镜像下载模型，避免卡住
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 设置中文字体，避免中文显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 1. Cityscapes 的 19 个类别
# ------------------------------------------------------------
CITYSCAPES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# 参与异质性计算的五个指标（色调是 RGB，不参与方差计算）
METRICS = ["greenness", "openness", "enclosure", "walkability", "complexity"]

# ------------------------------------------------------------
# 2. 两个可调参数：两个尺度各看多少张邻居
# ------------------------------------------------------------
WINDOW_IH = 1   # 即时异质性：前后各 1 张
WINDOW_CH = 3   # 背景异质性：前后各 3 张


# ------------------------------------------------------------
# 3. 算六个指标（和第二步一样的逻辑）
# ------------------------------------------------------------
def compute_indicators(pred):
    """输入分割结果，输出五个数值指标（色调另外算）"""
    total = pred.size

    # 每个类别占总像素的比例
    props = {}
    for i, name in enumerate(CITYSCAPES):
        props[name] = (pred == i).sum() / total

    # 指标 1：绿视率
    greenness = props["vegetation"]

    # 指标 2：开敞度
    openness = props["sky"]

    # 指标 3：围合度
    vertical = (
        props["building"] + props["wall"] + props["fence"]
        + props["pole"] + props["vegetation"]
    )
    enclosure = vertical / (vertical + props["sky"]) if (vertical + props["sky"]) > 0 else 0

    # 指标 4：可步行性
    # 注意：分子里含"行人"，当分母（道路面积）很小时，结果可能远大于 1。
    # 用 min(1.0, ...) 把它限制在 0-1 之间——可步行性是个比例，不该超过 1。
    walk_infra = props["sidewalk"] + props["person"]
    walk_total = props["road"] + props["sidewalk"]
    walkability = min(1.0, walk_infra / walk_total) if walk_total > 0 else 0

    # 指标 5：复杂度（香农熵，归一化到 0-1）
    p = np.array(list(props.values()))
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    complexity = entropy / np.log(len(CITYSCAPES))

    return {
        "greenness": greenness,
        "openness": openness,
        "enclosure": enclosure,
        "walkability": walkability,
        "complexity": complexity,
    }


# ------------------------------------------------------------
# 4. 核心函数：算某一张图的异质性
# ------------------------------------------------------------
def compute_heterogeneity(all_indicators, center_idx, window):
    """
    all_indicators: 所有图的指标列表（每个元素是一个字典）
    center_idx:    要算哪张图（它的下标）
    window:        看前后各多少张

    返回: (各指标的方差字典, 平均后的异质性分数)
    """
    # 确定邻居的下标范围（注意不能越界）
    start = max(0, center_idx - window)
    end = min(len(all_indicators), center_idx + window + 1)

    # 收集邻居的指标值（跳过它自己）
    neighbor_values = {m: [] for m in METRICS}
    for i in range(start, end):
        if i == center_idx:
            continue  # 不把自己算进去
        for m in METRICS:
            neighbor_values[m].append(all_indicators[i][m])

    # 对每个指标算方差
    variances = {}
    for m in METRICS:
        vals = neighbor_values[m]
        # 至少要有 2 个邻居才能算方差，否则记 0
        variances[m] = np.var(vals) if len(vals) >= 2 else 0.0

    # 把五个指标的方差取平均，作为这张图的"异质性分数"
    heterogeneity = float(np.mean(list(variances.values())))

    return variances, heterogeneity


# ------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------
def main():
    # 5.1 加载模型
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    print("正在加载模型……")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    print("模型加载完成。\n")

    # 5.2 找出图片（按文件名排序 —— 这里假设图片顺序 = 沿街行走的顺序）
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)

    image_paths = sorted(
        glob(os.path.join(image_dir, "*.jpg"))
        + glob(os.path.join(image_dir, "*.jpeg"))
        + glob(os.path.join(image_dir, "*.png"))
    )

    if len(image_paths) < 4:
        print(f"【提示】异质性计算至少需要 4 张图片，现在只有 {len(image_paths)} 张。")
        print(f"请把同一条街上拍的街景图，按顺序放进：{os.path.abspath(image_dir)}")
        print("文件名建议带编号，例如 street_01.jpg、street_02.jpg，这样顺序才正确。")
        return

    print(f"共找到 {len(image_paths)} 张图片，开始逐张处理……\n")

    # 5.3 第一遍：先把每张图的指标算出来
    all_indicators = []
    for path in image_paths:
        filename = os.path.basename(path)
        image = Image.open(path).convert("RGB")

        # 分割
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].numpy()

        # 算指标
        ind = compute_indicators(pred)
        ind["image"] = filename
        all_indicators.append(ind)
        print(f"  已算指标：{filename}")

    # 5.4 第二遍：对每张图算两个尺度的异质性
    print("\n开始计算异质性……")
    rows = []
    for i, ind in enumerate(all_indicators):
        # 即时异质性（看前后各 1 张）
        _, ih = compute_heterogeneity(all_indicators, i, WINDOW_IH)
        # 背景异质性（看前后各 3 张）
        _, ch = compute_heterogeneity(all_indicators, i, WINDOW_CH)

        rows.append({
            "image": ind["image"],
            "IH": round(ih, 6),
            "CH": round(ch, 6),
        })

    # 5.5 输出表格
    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/heterogeneity.csv", index=False, encoding="utf-8-sig")
    print("\n结果表已保存到：outputs/heterogeneity.csv")
    print(df.to_string(index=False))

    # 5.6 画图：IH 和 CH 随图片顺序的变化
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(df)), df["IH"], marker="o", label="IH 即时异质性")
    plt.plot(range(len(df)), df["CH"], marker="s", label="CH 背景异质性")
    plt.title("视觉异质性沿街道序列的变化")
    plt.xlabel("图片顺序")
    plt.ylabel("异质性（方差）")
    plt.xticks(range(len(df)), df["image"], rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/fig3_heterogeneity.png", dpi=150, bbox_inches="tight")
    print("\n异质性曲线图已保存到：outputs/fig3_heterogeneity.png")

    # 5.7 顺便说一句最有意思的结论
    print("\n【小发现】")
    print(f"  异质性最高的位置：{df.loc[df['CH'].idxmax(), 'image']}（CH = {df['CH'].max():.6f}）")
    print(f"  异质性最低的位置：{df.loc[df['CH'].idxmin(), 'image']}（CH = {df['CH'].min():.6f}）")
    print("  异质性高 = 这一段街景变化丰富；低 = 这一段比较单调。")


# 只有直接运行这个文件时，才会执行 main()
if __name__ == "__main__":
    main()
