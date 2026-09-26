from pathlib import Path
import io

from PIL import Image
import pytest
import requests

API_URL = "http://127.0.0.1:8000/predict"

TEST_IMAGE = "test.jpg"


def test_predict_success():
    with open(TEST_IMAGE, "rb") as image_file:
        files = {
            "file": (
                "test.jpg",
                image_file,
                "image/jpeg"
            )
        }

        response = requests.post(
            API_URL,
            files=files
        )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["filename"] == "test.jpg"
    assert data["output_count"] == 10
    assert data["prediction_shape"] == [1, 1, 352, 352]

def test_predict_invalid_file():
    files = {
        "file": (
            "test.txt",
            b"this is not an image",
            "text/plain"
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid image file"

def test_predict_missing_file():
    response = requests.post(
        API_URL
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data

@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_predict_different_image_formats(image_format):
    # 临时创建一张 RGB 图片
    image = Image.new("RGB", (352, 352), color="white")

    # 将图片保存到内存，而不是保存到磁盘
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)

    # 根据图片格式设置文件名和 Content-Type
    extension = "jpg" if image_format == "JPEG" else "png"
    content_type = "image/jpeg" if image_format == "JPEG" else "image/png"

    files = {
        "file": (
            f"test.{extension}",
            image_bytes,
            content_type
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["output_count"] == 10
    assert data["prediction_shape"] == [1, 1, 352, 352]

@pytest.mark.parametrize("image_size", [
    (32, 32),
    (352, 352),
    (1920, 1080),
])
def test_predict_different_image_sizes(image_size):
    image = Image.new("RGB", image_size, color="white")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    files = {
        "file": (
            "test.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["output_count"] == 10
    assert data["prediction_shape"] == [1, 1, 352, 352]

def test_predict_corrupted_image():
    files = {
        "file": (
            "fake.jpg",
            b"This is a corrupted JPEG file",
            "image/jpeg"
        )
    }

    response = requests.post(
        API_URL,
        files=files
    )

    assert response.status_code == 400

    data = response.json()

    assert data["detail"] == "Invalid image file"

def test_predict_response_schema():
    with open(TEST_IMAGE, "rb") as image_file:
        files = {
            "file": (
                "test.jpg",
                image_file,
                "image/jpeg"
            )
        }

        response = requests.post(
            API_URL,
            files=files
        )

    assert response.status_code == 200

    data = response.json()

    # 检查字段是否存在
    expected_fields = [
        "success",
        "filename",
        "output_count",
        "prediction_shape",
        "result_path",
        "message"
    ]

    for field in expected_fields:
        assert field in data

    # 检查字段类型
    assert isinstance(data["success"], bool)
    assert isinstance(data["filename"], str)
    assert isinstance(data["output_count"], int)
    assert isinstance(data["prediction_shape"], list)
    assert isinstance(data["result_path"], str)
    assert isinstance(data["message"], str)
