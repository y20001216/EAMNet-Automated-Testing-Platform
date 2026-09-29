from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_structure():
    """检查项目核心目录和文件是否存在。"""

    required_paths = [
        PROJECT_ROOT / "app",
        PROJECT_ROOT / "lib",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "tests" / "data" / "golden",
        PROJECT_ROOT / "tests" / "baselines",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "test.jpg",
    ]

    for path in required_paths:
        assert path.exists(), f"缺少项目文件或目录：{path}"


def test_golden_samples():
    """检查 Golden Sample 是否完整。"""

    golden_dir = PROJECT_ROOT / "tests" / "data" / "golden"

    images = list(golden_dir.glob("*.jpg"))

    assert len(images) > 0, "Golden Sample 不存在"

    for image in images:
        gt = golden_dir / f"{image.stem}.png"
        assert gt.exists(), f"缺少 {image.name} 对应的 GT：{gt.name}"


def test_baseline_files():
    """检查模型质量和性能 baseline 是否存在。"""

    baseline_dir = PROJECT_ROOT / "tests" / "baselines"

    assert (baseline_dir / "mae_baseline.json").exists()
    assert (baseline_dir / "performance_baseline.json").exists()