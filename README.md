# 面向深度学习模型的自动化测试与质量评估平台

## 1. 项目简介

本项目面向计算机视觉深度学习模型推理服务，构建一套自动化测试与质量评估平台，用于验证模型服务的功能正确性、接口稳定性、Web 交互流程、模型预测质量以及系统性能。

项目以 Pytest 为核心测试框架，将 API 自动化测试、Web UI 自动化测试、模型质量回归测试和性能测试进行整合，并结合 Allure、Docker 和 GitHub Actions 构建自动化测试与持续集成环境。

项目主要用于验证深度学习模型在不同版本、不同输入数据和不同运行条件下的稳定性，为模型服务的持续迭代提供质量保障。

---

## 2. 技术栈

| 技术             | 用途                 |
| -------------- | ------------------ |
| Python         | 测试脚本及项目开发          |
| Pytest         | 自动化测试框架            |
| Requests       | REST API 自动化测试     |
| Playwright     | Web UI / E2E 自动化测试 |
| PyTorch        | 深度学习模型推理与质量测试      |
| Locust         | 接口性能与并发测试          |
| Allure         | 测试报告生成             |
| Docker         | 测试环境容器化            |
| GitHub Actions | 持续集成与自动化测试         |

---

## 3. 测试体系

项目按照不同测试层次构建自动化测试体系。

### 3.1 API 自动化测试

基于 Pytest + Requests 对模型推理接口进行自动化验证。

测试场景包括：

* 正常请求
* 参数异常
* 边界条件
* 文件类型异常
* 缺少必要参数
* 接口返回结果校验

通过统一的测试数据和断言逻辑，提高测试用例的可维护性。

---

### 3.2 Web UI 自动化测试

基于 Playwright 对 Web 端主要业务流程进行端到端测试。

覆盖：

* 图片上传
* 模型检测
* 批量检测
* 检测结果查看
* 异常输入处理
* 页面截图及测试 Trace

用于验证前端页面与后端模型服务之间的完整业务流程。

---

### 3.3 模型质量回归测试

针对深度学习模型输出结果进行自动化质量评估。

使用 Golden Sample 与 Ground Truth 构建模型回归测试数据集，并通过质量指标对模型版本进行比较。

目前主要使用：

* MAE
* IoU
* F-measure

同时保存历史质量基线，用于发现模型更新后可能出现的性能退化。

---

### 3.4 性能测试

基于 Locust 对模型服务接口进行并发性能测试。

主要关注：

* 请求响应时间
* P50 延迟
* P95 延迟
* P99 延迟
* 吞吐量
* 错误率

通过性能基线对不同版本进行对比，为模型服务性能优化提供数据支持。

---

## 4. 项目结构

```text
TESTProject/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions CI
│
├── app/
│   └── main.py                   # 服务端代码
│
├── lib/                          # 模型相关代码
│
├── tests/
│   ├── baselines/                # 测试基线
│   │   ├── mae_baseline.json
│   │   └── performance_baseline.json
│   │
│   ├── data/
│   │   └── golden/               # Golden Sample 测试数据
│   │
│   ├── test_api.py               # API 自动化测试
│   ├── test_camo_quality.py      # 模型质量测试
│   ├── test_eamnet.py            # 模型相关测试
│   ├── test_eamnet_performance.py# 性能测试
│   ├── test_playwright.py        # Web UI 自动化测试
│   └── test_project_smoke.py     # 项目 Smoke Test
│
├── docs/
│   └── locust_test_report.md     # 性能测试报告
│
├── Dockerfile                    # Docker 测试环境
├── .dockerignore                 # Docker 构建忽略文件
├── requirements.txt              # Python 依赖
├── generate_baseline.py          # Baseline 生成
└── README.md
```

---

## 5. 本地运行

### 5.1 安装依赖

建议使用 Python 虚拟环境：

```bash
pip install -r requirements.txt
```

---

### 5.2 执行 Smoke Test

```bash
pytest tests/test_project_smoke.py -v
```

Smoke Test 用于验证项目核心目录、Golden Sample 和测试 Baseline 是否完整。

示例结果：

```text
============================= test session starts ==============================

tests/test_project_smoke.py::test_project_structure PASSED
tests/test_project_smoke.py::test_golden_samples PASSED
tests/test_project_smoke.py::test_baseline_files PASSED

============================== 3 passed ==============================
```

---

## 6. Docker 运行

项目提供 Docker 测试环境，用于统一 Python 和测试依赖环境。

### 构建 Docker 镜像

```bash
docker build -t eamnet-test .
```

### 执行 Smoke Test

```bash
docker run --rm eamnet-test
```

容器启动后会自动执行：

```bash
pytest tests/test_project_smoke.py -v
```

当前 Docker 环境已验证 Smoke Test：

```text
3 passed
```

---

## 7. GitHub Actions

项目使用 GitHub Actions 进行持续集成。

每次向 `main` 分支 Push 或提交 Pull Request 时，自动执行：

```text
Checkout Repository
        ↓
Setup Python 3.8
        ↓
Install Dependencies
        ↓
Run Smoke Test
        ↓
Test Result
```

当前 CI 主要用于验证项目基础结构和测试资源是否完整。

---

## 8. 测试数据与模型权重

项目使用 Golden Sample 和 Ground Truth 作为模型质量回归测试数据。

模型权重文件体积较大，因此未直接提交至 Git 仓库。

`Git` 中通过 `.gitignore` 忽略模型训练输出及权重文件，例如：

```text
snapshot/
*.pth
```

这样可以避免大型模型文件影响 Git 仓库体积。

---

## 9. 项目特点

本项目将传统软件测试方法与深度学习模型质量评估结合，形成多层次自动化测试体系：

```text
                 自动化测试平台
                       │
       ┌───────────────┼───────────────┐
       │               │               │
    API测试         UI测试          模型质量测试
       │               │               │
    Requests        Playwright       PyTorch
       │               │               │
       └───────────────┼───────────────┘
                       │
                    性能测试
                       │
                    Locust
                       │
              ┌────────┴────────┐
              │                 │
           Allure          GitHub Actions
              │                 │
              └────────┬────────┘
                       │
                    Docker
```

---

## 10. 后续计划

* 完善 API、UI 和模型质量测试用例
* 增加测试数据管理与参数化能力
* 完善 Allure 测试报告
* 优化性能测试基线
* 根据实际模型服务部署方式完善 CI 流程
* 持续补充异常场景和回归测试用例
