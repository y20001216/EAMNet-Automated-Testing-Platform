import sys
import os
from pathlib import Path

# 让 pytest 能找到项目根目录下的 lib、utils 等模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from PIL import Image
from torchvision import transforms

from lib.EAMNet import Network


# =========================
# 路径配置
# =========================

TEST_IMAGE = PROJECT_ROOT / "test.jpg"
CHECKPOINT = PROJECT_ROOT / "snapshot" / "EAMNet_E_b4_3" / "Net_epoch_best.pth"

TEST_SIZE = 352

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Model Fixture
# =========================

@pytest.fixture(scope="session")
def model():
    """
    整个 pytest 测试会话只加载一次 EAMNet。
    """

    model = Network(imagenet_pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# =========================
# Input Fixture
# =========================

@pytest.fixture(scope="session")
def input_tensor():
    image = Image.open(TEST_IMAGE).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((TEST_SIZE, TEST_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    return image.to(DEVICE)


@pytest.fixture(scope="session")
def model_outputs(model, input_tensor):
    """
    运行 EAMNet 一次，并保存模型输出。
    后续测试共享这一次推理结果。
    """
    with torch.no_grad():
        outputs = model(input_tensor)

    return outputs


# =========================
# Tests
# =========================

def test_checkpoint_exists():
    """检查模型权重文件是否存在。"""

    assert CHECKPOINT.exists()


def test_eamnet_inference(model_outputs):
    """检查 EAMNet 是否能够正常完成推理。"""

    assert model_outputs is not None


def test_eamnet_output_count(model_outputs):
    """检查模型输出数量是否符合预期。"""

    assert len(model_outputs) == 10


def test_main_prediction_shape(model_outputs):
    """检查主预测结果的 Batch 和 Channel 维度。"""

    main_prediction = model_outputs[1]

    assert main_prediction.shape[0] == 1
    assert main_prediction.shape[1] == 1


def test_main_prediction_not_nan(model_outputs):
    """检查主预测结果是否存在 NaN。"""

    main_prediction = model_outputs[1]

    assert not torch.isnan(main_prediction).any()


def test_main_prediction_not_inf(model_outputs):
    """检查主预测结果是否存在 Inf。"""

    main_prediction = model_outputs[1]

    assert not torch.isinf(main_prediction).any()


def test_main_prediction_probability_range(model_outputs):
    """检查主预测经过 sigmoid 后是否处于 [0, 1] 范围。"""

    main_prediction = model_outputs[1]

    probability = torch.sigmoid(main_prediction)

    assert probability.min() >= 0.0
    assert probability.max() <= 1.0
