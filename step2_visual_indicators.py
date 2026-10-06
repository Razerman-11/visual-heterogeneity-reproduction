# ============================================================
# 复现论文第二步：从语义分割结果，算出六个视觉指标
# 对应论文《Beyond single snapshots》的 3.1 节（Table 1）
#
# 【这一步在做什么】
# 第一步得到的是"分割结果"——一张图上每个像素被标了类别。
# 这一步要把那堆类别，变成六个"数字指标"。
#
# 【输入】images 文件夹里的街景图
# 【输出】一张表（indicators.csv）：每张图一行，六个指标六列
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
from sklearn.cluster import KMeans
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 设置中文字体，避免中文显示成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 1. Cityscapes 的 19 个类别（顺序不能改，编号是固定的）
# ------------------------------------------------------------
CITYSCAPES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


# ------------------------------------------------------------
# 2. 核心函数：输入分割结果，输出六个指标
# ------------------------------------------------------------
def compute_indicators(pred):
    """
    pred: 分割结果，形状是 (高, 宽)，每个值 0-18，代表那个像素是什么类别
    返回: 一个字典，装着六个指标
    """
    # 总像素数 = 高 × 宽
    total = pred.size

    # 先算出每个类别占总像素的比例，存成字典
    # 例如 props["vegetation"] 就是"植被占了画面的百分之多少"
    props = {}
    for i, name in enumerate(CITYSCAPES):
        # pred == i 生成一个布尔数组（哪些像素属于第 i 类）
        # .sum() 数出 True 的个数，再除以总数就是比例
        props[name] = (pred == i).sum() / total

    # ---- 指标 1：绿视率 greenness ----
    # 定义：植被像素占总像素的比例（论文里的 GVI）
    greenness = props["vegetation"]

    # ---- 指标 2：开敞度 openness ----
    # 定义：天空像素占总像素的比例
    openness = props["sky"]

    # ---- 指标 3：围合度 enclosure ----
    # 定义：垂直元素 / (垂直元素 + 天空)
    # "垂直元素"指立在视野里的东西：建筑、墙、栅栏、杆、树
    vertical = (
        props["building"] + props["wall"] + props["fence"]
        + props["pole"] + props["vegetation"]
    )
    # 加一个判断，避免分母为 0 时报错
    enclosure = vertical / (vertical + props["sky"]) if (vertical + props["sky"]) > 0 else 0

    # ---- 指标 4：可步行性 walkability ----
    # 定义：步行设施 / 道路总面积
    # 步行设施 = 人行道 + 行人；道路总面积 = 车行道 + 人行道
    # 注意：分子里含"行人"，当分母很小时结果可能远大于 1，
    # 所以用 min(1.0, ...) 把它限制在 0-1 之间（比例指标不该超过 1）
    walk_infra = props["sidewalk"] + props["person"]
    walk_total = props["road"] + props["sidewalk"]
    walkability = min(1.0, walk_infra / walk_total) if walk_total > 0 else 0

    # ---- 指标 5：复杂度 complexity ----
    # 定义：类别分布的"香农熵"，再归一化到 0-1
    # 直觉：一张图里类别越多、分布越均匀，信息量越大（越复杂）
    #       一张图几乎只有天空，信息量就小（越单调）
    p = np.array(list(props.values()))  # 把 19 个比例转成数组
    p = p[p > 0]                        # 去掉 0，因为 log(0) 没有定义
    entropy = -(p * np.log(p)).sum()    # 香农熵公式
    complexity = entropy / np.log(len(CITYSCAPES))  # 归一化到 0-1

    return {
        "greenness": greenness,
        "openness": openness,
        "enclosure": enclosure,
        "walkability": walkability,
        "complexity": complexity,
    }


# ------------------------------------------------------------
# 3. 算色调：用 K-means 找出图片的主色
# ------------------------------------------------------------
def compute_color_tone(image, k=3):
    """
    image: PIL 图片
    k: 把颜色聚成几类
    返回: 主色的 RGB（三个数字）
    """
    # 先缩小图片，加速计算（色调不需要原始分辨率）
    small = image.resize((120, 120))
    # 把图片转成 (像素数, 3) 的数组，每行是一个像素的 RGB
    pixels = np.array(small).reshape(-1, 3)

    # K-means 聚类：把像素颜色分成 k 组
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pixels)

    # 找出"包含像素最多"的那一组，它的中心色就是主色
    counts = np.bincount(km.labels_)
    main_color = km.cluster_centers_[counts.argmax()]

    return main_color  # [R, G, B]


# ------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------
def main():
    # 4.1 加载预训练模型
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    print("正在加载模型……")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    print("模型加载完成。\n")

    # 4.2 找出要处理的图片
    # 约定：把所有街景图放进 images 文件夹
    image_dir = "images"
    os.makedirs(image_dir, exist_ok=True)  # 没有这个文件夹就自动创建一个

    image_paths = sorted(
        glob(os.path.join(image_dir, "*.jpg"))
        + glob(os.path.join(image_dir, "*.jpeg"))
        + glob(os.path.join(image_dir, "*.png"))
    )

    if not image_paths:
        print(f"【提示】在 {image_dir} 文件夹里没找到图片。")
        print(f"请把你的街景图复制到这个文件夹：{os.path.abspath(image_dir)}")
        return

    print(f"共找到 {len(image_paths)} 张图片，开始逐张处理……\n")

    # 4.3 逐张处理
    rows = []  # 存放每张图的结果
    for path in image_paths:
        filename = os.path.basename(path)

        # 读图
        image = Image.open(path).convert("RGB")

        # 分割
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        # 放大回原图尺寸，然后每个像素取概率最大的类别
        logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].numpy()

        # 算六个指标
        row = compute_indicators(pred)
        r, g, b = compute_color_tone(image)
        row["color_r"] = round(float(r))
        row["color_g"] = round(float(g))
        row["color_b"] = round(float(b))
        row["image"] = filename

        rows.append(row)
        print(f"  已处理：{filename}")

    # 4.4 汇总成表格
    df = pd.DataFrame(rows)
    # 把列排成好看的顺序
    df = df[[
        "image", "greenness", "openness", "enclosure",
        "walkability", "complexity", "color_r", "color_g", "color_b",
    ]]

    # 保存成 CSV（用 utf-8-sig 编码，这样用 Excel 打开不会乱码）
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/indicators.csv", index=False, encoding="utf-8-sig")
    print("\n结果表已保存到：outputs/indicators.csv")
    print(df.to_string(index=False))

    # 4.5 画一张柱状图，直观看看各个指标
    # 只画前五个数值指标（色调是 RGB 三个数，不适合画柱状图）
    metrics = ["greenness", "openness", "enclosure", "walkability", "complexity"]
    df[metrics].plot(kind="bar", figsize=(12, 5))
    plt.title("各街景图的视觉指标")
    plt.ylabel("数值（0-1）")
    plt.xticks(range(len(df)), df["image"], rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/fig2_indicators.png", dpi=150, bbox_inches="tight")
    print("\n指标对比图已保存到：outputs/fig2_indicators.png")


# 只有直接运行这个文件时，才会执行 main()
if __name__ == "__main__":
    main()
