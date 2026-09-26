import sys
import json
from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import allure


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.EAMNet import Network


# ============================================================
# 2. 测试数据和模型
# ============================================================

GOLDEN_DIR = PROJECT_ROOT / "tests" / "data" / "golden"

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
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. pytest fixture：加载一次模型
# ============================================================

@pytest.fixture(scope="session")
def model():

    model = Network(imagenet_pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 4. pytest fixture：读取 MAE baseline
# ============================================================

@pytest.fixture(scope="session")
def mae_baselines():
    """
    从 JSON 文件读取 Golden Sample 的 MAE baseline。
    """

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        baselines = json.load(f)

    return baselines


# ============================================================
# 5. 图片预处理
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
# 6. 计算 MAE
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

    mae = np.mean(
        np.abs(
            prediction - gt
        )
    )

    return mae

def calculate_f_measure(
    prediction,
    gt,
    beta2=0.3
):
    """
    计算最大 F-measure。
    """

    precision = []
    recall = []

    for threshold in np.linspace(0, 1, 256):

        binary_prediction = (
            prediction >= threshold
        ).astype(np.float32)

        true_positive = (
            binary_prediction * gt
        ).sum()

        pre = (
            true_positive
            / (binary_prediction.sum() + 1e-8)
        )

        rec = (
            true_positive
            / (gt.sum() + 1e-8)
        )

        precision.append(pre)
        recall.append(rec)

    precision = np.array(precision)
    recall = np.array(recall)

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
# 7. 单张图片推理并计算 MAE + F-measure
# ============================================================

def predict_metrics(
    model,
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

    # EAMNet 推理
    with torch.no_grad():

        outputs = model(image)

    # 主预测结果
    prediction = outputs[1]

    # 转换成 0~1 概率
    prediction = torch.sigmoid(
        prediction
    )

    # 恢复到 GT 原始尺寸
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

    # 计算最大 F-measure
    f_measure = calculate_f_measure(
        prediction,
        gt_array
    )

    return mae, f_measure


# ============================================================
# 8. Golden Sample MAE 结果
# ============================================================

@pytest.fixture(scope="session")
def camo_results(model):
    """
    自动发现 Golden Sample 文件夹中的所有 JPG 图片，
    并计算每张图片的 MAE 和 F-measure。
    """

    results = {}

    image_files = sorted(
        GOLDEN_DIR.glob("*.jpg")
    )

    for image_path in image_files:

        image_name = image_path.name

        gt_path = (
            GOLDEN_DIR
            / (image_path.stem + ".png")
        )

        assert gt_path.exists(), (
            f"Missing GT file for {image_name}"
        )

        mae, f_measure = predict_metrics(
            model,
            image_path,
            gt_path
        )

        results[image_name] = {
            "mae": mae,
            "f_measure": f_measure
        }

    return results

# ============================================================
# 9. CAMO Golden Sample 单张图片回归测试
# ============================================================

def test_camo_mae_regression(
    camo_results,
    mae_baselines
):

    for image_name, metrics in camo_results.items():

        mae = metrics["mae"]
        f_measure = metrics["f_measure"]

        # ============================
        # 当前图片 baseline
        # ============================

        with allure.step(f"读取 {image_name} 的 baseline"):

            baseline_mae = (
                mae_baselines[image_name]["mae"]
            )

            baseline_f_measure = (
                mae_baselines[image_name]["f_measure"]
            )

        # ============================
        # 计算允许范围
        # ============================

        with allure.step(f"计算 {image_name} 的允许范围"):

            max_allowed_mae = (
                baseline_mae * 1.20
            )

            min_allowed_f_measure = (
                baseline_f_measure * 0.90
            )

        # ============================
        # 输出结果
        # ============================

        print()

        print(
            f"Image: {image_name}"
        )

        print(
            f"Baseline MAE: "
            f"{baseline_mae:.6f}"
        )

        print(
            f"Current MAE:  "
            f"{mae:.6f}"
        )

        print(
            f"Max allowed:  "
            f"{max_allowed_mae:.6f}"
        )

        print(
            f"Baseline F:    "
            f"{baseline_f_measure:.6f}"
        )

        print(
            f"Current F:     "
            f"{f_measure:.6f}"
        )

        print(
            f"Min allowed F: "
            f"{min_allowed_f_measure:.6f}"
        )

        # ============================
        # MAE 回归判断
        # ============================

        with allure.step(
            f"{image_name} MAE 回归判断"
        ):

            assert mae <= max_allowed_mae

        # ============================
        # F-measure 回归判断
        # ============================

        with allure.step(
            f"{image_name} F-measure 回归判断"
        ):

            assert f_measure >= min_allowed_f_measure


# ============================================================
# 10. CAMO Golden Sample 平均指标回归测试
# ============================================================

def test_camo_average_metrics(
    camo_results,
    mae_baselines
):

    # ============================
    # 当前平均 MAE
    # ============================

    with allure.step(
        "统计当前 Golden Sample 平均 MAE"
    ):

        maes = [
            metrics["mae"]
            for metrics in camo_results.values()
        ]

        average_mae = np.mean(
            maes
        )

    # ============================
    # 当前平均 F-measure
    # ============================

    with allure.step(
        "统计当前 Golden Sample 平均 F-measure"
    ):

        f_measures = [
            metrics["f_measure"]
            for metrics in camo_results.values()
        ]

        average_f_measure = np.mean(
            f_measures
        )

    # ============================
    # 读取 baseline
    # ============================

    with allure.step(
        "读取平均指标 baseline"
    ):

        baseline_average_mae = (
            mae_baselines["average"]["mae"]
        )

        baseline_average_f_measure = (
            mae_baselines["average"]["f_measure"]
        )

    # ============================
    # 计算允许范围
    # ============================

    with allure.step(
        "计算平均指标允许范围"
    ):

        max_allowed_mae = (
            baseline_average_mae * 1.20
        )

        min_allowed_f_measure = (
            baseline_average_f_measure * 0.90
        )

    # ============================
    # 输出结果
    # ============================

    print()

    print("=" * 60)

    print(
        "CAMO Average Quality Regression Test"
    )

    print("=" * 60)

    print(
        f"Baseline Average MAE: "
        f"{baseline_average_mae:.6f}"
    )

    print(
        f"Current Average MAE:  "
        f"{average_mae:.6f}"
    )

    print(
        f"Max Allowed MAE:      "
        f"{max_allowed_mae:.6f}"
    )

    print()

    print(
        f"Baseline Average F:   "
        f"{baseline_average_f_measure:.6f}"
    )

    print(
        f"Current Average F:    "
        f"{average_f_measure:.6f}"
    )

    print(
        f"Min Allowed F:        "
        f"{min_allowed_f_measure:.6f}"
    )

    # ============================
    # 平均 MAE 回归
    # ============================

    with allure.step(
        "执行平均 MAE 回归判断"
    ):

        assert average_mae <= max_allowed_mae

    # ============================
    # 平均 F-measure 回归
    # ============================

    with allure.step(
        "执行平均 F-measure 回归判断"
    ):

        assert average_f_measure >= min_allowed_f_measure