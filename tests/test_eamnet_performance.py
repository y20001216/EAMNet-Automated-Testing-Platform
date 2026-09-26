import sys
import time
import json
import statistics
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms



# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from lib.EAMNet import Network


# ============================================================
# 2. 测试配置
# ============================================================

TEST_IMAGE = PROJECT_ROOT / "test.jpg"

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
# 3. 加载模型
# ============================================================

@pytest.fixture(scope="session")
def model():

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

    return model


# ============================================================
# 4. 准备测试图片
# ============================================================

@pytest.fixture(scope="session")
def input_tensor():

    image = Image.open(
        TEST_IMAGE
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
# 5. 测试模型推理时间
# ============================================================

def test_eamnet_inference_time(
    model,
    input_tensor
):

    # ========================================================
    # 1. 模型预热
    # ========================================================

    warmup_rounds = 3

    for _ in range(warmup_rounds):

        with torch.no_grad():
            model(input_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


    # ========================================================
    # 2. 正式测试
    # ========================================================

    test_rounds = 10

    inference_times = []

    for i in range(test_rounds):

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = (
            end_time - start_time
        )

        inference_times.append(
            inference_time
        )


    # ========================================================
    # 3. 统计结果
    # ========================================================

    average_time = sum(
        inference_times
    ) / len(inference_times)

    median_time = statistics.median(
        inference_times
    )

    min_time = min(
        inference_times
    )

    max_time = max(
        inference_times
    )


    # ========================================================
    # 4. 输出测试结果
    # ========================================================

    print()

    print("=" * 60)

    print(
        "EAMNet Inference Performance Test"
    )

    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Warm-up rounds: {warmup_rounds}"
    )

    print(
        f"Test rounds: {test_rounds}"
    )

    print()

    for i, t in enumerate(
        inference_times,
        start=1
    ):

        print(
            f"Round {i}: "
            f"{t:.6f} s"
        )

    print()

    print(
        f"Average: "
        f"{average_time:.6f} s"
    )

    print(
        f"Median:  "
        f"{median_time:.6f} s"
    )

    print(
        f"Min:     "
        f"{min_time:.6f} s"
    )

    print(
        f"Max:     "
        f"{max_time:.6f} s"
    )

    print("=" * 60)

    # ========================================================
    # 5. 读取性能 baseline
    # ========================================================

    BASELINE_FILE = (
            PROJECT_ROOT
            / "tests"
            / "baselines"
            / "performance_baseline.json"
    )

    with open(
            BASELINE_FILE,
            "r",
            encoding="utf-8"
    ) as f:

        baseline = json.load(f)

    baseline_average_time = (
        baseline["average_inference_time"]
    )

    # ========================================================
    # 6. 计算允许的最大推理时间
    # ========================================================

    allowed_increase = 0.20

    max_allowed_time = (
            baseline_average_time
            * (1 + allowed_increase)
    )

    # ========================================================
    # 7. 输出性能回归结果
    # ========================================================

    print()

    print(
        f"Baseline Average: "
        f"{baseline_average_time:.6f} s"
    )

    print(
        f"Current Average:  "
        f"{average_time:.6f} s"
    )

    print(
        f"Max Allowed:      "
        f"{max_allowed_time:.6f} s"
    )

    # ========================================================
    # 8. 性能回归判断
    # ========================================================

    assert average_time <= max_allowed_time