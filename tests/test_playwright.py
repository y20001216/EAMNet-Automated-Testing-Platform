from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://localhost:8501")
        page.wait_for_timeout(3000)

        yield page

        browser.close()

@pytest.fixture
def single_image_page(page):
    # 点击“单图检测”
    page.get_by_text("单图检测", exact=True).click()

    # 等待单图识别页面加载
    page.wait_for_timeout(2000)

    return page

def test_upload_image(single_image_page):
    # 定位“上传单张图片”的文件上传框
    file_input = single_image_page.get_by_label("上传单张图片").get_by_test_id(
        "stFileUploaderDropzoneInput"
    )

    # 确认上传框存在
    assert file_input.count() == 1

    # 上传测试图片
    file_input.set_input_files("test.jpg")

    # 等待上传完成
    single_image_page.wait_for_timeout(2000)

    # 点击“开始检测”
    single_image_page.get_by_text("开始检测", exact=True).click()

    # 等待检测完成
    single_image_page.get_by_text(
        "检测完成", exact=False
    ).wait_for(timeout=30000)

    # 等待结果区域完成渲染
    single_image_page.wait_for_timeout(3000)

    # 获取页面文本
    page_text = single_image_page.locator("body").inner_text()

    # 验证检测成功
    assert "检测完成" in page_text

    # 验证结果导出功能
    assert "结果导出" in page_text
    assert "下载 Mask" in page_text
    assert "下载 Overlay" in page_text

    # 验证性能信息
    assert "性能指标记录" in page_text
    assert "cuda" in page_text

    # =========================
    # 测试下载 Mask
    # =========================

    with single_image_page.expect_download() as download_info:
        single_image_page.get_by_text("下载 Mask", exact=True).click()

    download = download_info.value

    # 获取下载文件名
    file_name = download.suggested_filename

    print("下载文件名：", file_name)

    # 获取下载文件路径
    download_path = download.path()

    # 确认文件确实产生
    assert download_path is not None

    # 检查文件大小
    file_size = Path(download_path).stat().st_size

    print("下载文件大小：", file_size, "bytes")

    assert file_size > 0

def test_start_detection_without_upload(single_image_page):
    # 查找“开始检测”按钮
    start_button = single_image_page.get_by_text(
        "开始检测", exact=True
    )

    # 没有上传图片时，不应该出现“开始检测”按钮
    assert start_button.count() == 0

    print("\n未上传图片时，“开始检测”按钮不存在")

def test_page_with_fixture(page):
        print("页面标题：", page.title())

        assert page.title() == "GateReg COD 智能检测平台"

def test_download_overlay(single_image_page):
    # 定位“上传单张图片”的文件上传框
    file_input = single_image_page.get_by_label("上传单张图片").get_by_test_id(
        "stFileUploaderDropzoneInput"
    )

    # 确认上传框存在
    assert file_input.count() == 1

    # 上传测试图片
    file_input.set_input_files("test.jpg")

    # 等待上传完成
    single_image_page.wait_for_timeout(2000)

    # 点击“开始检测”
    single_image_page.get_by_text("开始检测", exact=True).click()

    # 等待检测完成
    single_image_page.get_by_text(
        "检测完成", exact=False
    ).wait_for(timeout=30000)

    # 等待结果区域完成渲染
    single_image_page.wait_for_timeout(3000)

    # 获取页面文本
    page_text = single_image_page.locator("body").inner_text()

    # 确认 Overlay 下载功能存在
    assert "下载 Overlay" in page_text

    # =========================
    # 测试下载 Overlay
    # =========================

    with single_image_page.expect_download() as download_info:
        single_image_page.get_by_text(
            "下载 Overlay", exact=True
        ).click()

    download = download_info.value

    # 获取下载文件名
    file_name = download.suggested_filename

    print("Overlay 下载文件名：", file_name)

    # 获取下载文件路径
    download_path = download.path()

    # 确认文件确实产生
    assert download_path is not None

    # 检查文件大小
    file_size = Path(download_path).stat().st_size

    print("Overlay 下载文件大小：", file_size, "bytes")

    # 文件不能是空文件
    assert file_size > 0
