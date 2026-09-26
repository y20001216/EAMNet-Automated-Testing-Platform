import sys
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from lib.EAMNet import Network


# ============================================================
# 2. 配置
# ============================================================

GOLDEN_DIR = (
    PROJECT_ROOT
    / "tests"
    / "data"
    / "golden"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "tests"
    / "baselines"
    / "mae_baseline.json"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "snapshot"
    / "EAMNet_E_b4_3"
    / "Net_epoch_best.pth"
)

TEST_SIZE = 352

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. 加载模型
# ============================================================

print("Loading EAMNet model...")

model = Network(
    imagenet_pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)
model.eval()

print(
    f"Model loaded successfully."
)

print(
    f"Device: {DEVICE}"
)


# ============================================================
# 4. 图片预处理
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(
            (TEST_SIZE, TEST_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = transform(image)

    image = image.unsqueeze(0)

    return image.to(DEVICE)


# ============================================================
# 5. 计算 MAE
# ============================================================

def calculate_mae(
    prediction,
    gt
):

    prediction = prediction.astype(
        np.float32
    )

    gt = gt.astype(
        np.float32
    )

    return np.mean(
        np.abs(
            prediction - gt
        )
    )


# ============================================================
# 6. 计算 F-measure
# ============================================================

def calculate_f_measure(
    prediction,
    gt,
    beta2=0.3
):

    precision = []
    recall = []

    for threshold in np.linspace(
        0,
        1,
        256
    ):

        binary_prediction = (
            prediction >= threshold
        ).astype(np.float32)

        true_positive = (
            binary_prediction * gt
        ).sum()

        pre = (
            true_positive
            / (
                binary_prediction.sum()
                + 1e-8
            )
        )

        rec = (
            true_positive
            / (
                gt.sum()
                + 1e-8
            )
        )

        precision.append(pre)
        recall.append(rec)

    precision = np.array(
        precision
    )

    recall = np.array(
        recall
    )

    f_measure = (
        (1 + beta2)
        * precision
        * recall
        / (
            beta2 * precision
            + recall
            + 1e-8
        )
    )

    return f_measure.max()


# ============================================================
# 7. 单张图片计算指标
# ============================================================

def predict_metrics(
    image_path,
    gt_path
):

    # 读取 GT
    gt = Image.open(
        gt_path
    ).convert("L")

    gt_array = (
        np.array(gt).astype(
            np.float32
        ) / 255.0
    )

    # 图片预处理
    image = preprocess_image(
        image_path
    )

    # 模型推理
    with torch.no_grad():

        outputs = model(image)

    # 主预测结果
    prediction = outputs[1]

    # 转换为概率
    prediction = torch.sigmoid(
        prediction
    )

    # 恢复 GT 原始尺寸
    prediction = torch.nn.functional.interpolate(
        prediction,
        size=gt_array.shape,
        mode="bilinear",
        align_corners=False
    )

    # 转 numpy
    prediction = (
        prediction.cpu()
        .numpy()
        .squeeze()
    )

    # 计算 MAE
    mae = calculate_mae(
        prediction,
        gt_array
    )

    # 计算 F-measure
    f_measure = calculate_f_measure(
        prediction,
        gt_array
    )

    return mae, f_measure


# ============================================================
# 8. 自动发现 Golden Samples
# ============================================================

image_files = sorted(
    GOLDEN_DIR.glob("*.jpg")
)

print()
print(
    f"Found {len(image_files)} Golden Samples."
)

print()


# ============================================================
# 9. 计算所有 Golden Samples
# ============================================================

results = {}

maes = []
f_measures = []

for image_path in image_files:

    image_name = image_path.name

    gt_path = (
        GOLDEN_DIR
        / (
            image_path.stem
            + ".png"
        )
    )

    if not gt_path.exists():

        raise FileNotFoundError(
            f"Missing GT file: "
            f"{gt_path}"
        )

    mae, f_measure = predict_metrics(
        image_path,
        gt_path
    )

    results[image_name] = {
        "mae": round(
            float(mae),
            6
        ),
        "f_measure": round(
            float(f_measure),
            6
        )
    }

    maes.append(mae)

    f_measures.append(
        f_measure
    )

    print(
        f"{image_name}"
    )

    print(
        f"  MAE:        {mae:.6f}"
    )

    print(
        f"  F-measure:  {f_measure:.6f}"
    )


# ============================================================
# 10. 计算平均指标
# ============================================================

average_mae = np.mean(
    maes
)

average_f_measure = np.mean(
    f_measures
)

results["average"] = {
    "mae": round(
        float(average_mae),
        6
    ),
    "f_measure": round(
        float(average_f_measure),
        6
    )
}


# ============================================================
# 11. 保存 baseline
# ============================================================

BASELINE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    BASELINE_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 12. 输出结果
# ============================================================

print()
print("=" * 60)
print("Baseline Generation Complete")
print("=" * 60)

print(
    f"Average MAE:       "
    f"{average_mae:.6f}"
)

print(
    f"Average F-measure: "
    f"{average_f_measure:.6f}"
)

print()
print(
    f"Saved to:"
)

print(
    BASELINE_FILE
)