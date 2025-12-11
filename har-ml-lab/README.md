# HAR Machine Learning Lab 🚀

这是一个基于 **UCI HAR (Human Activity Recognition)** 数据集的机器学习实战项目。我们构建了一个完整的机器学习流水线，用于识别 6 种不同的人体动作（走路、上楼、下楼、坐、站、躺）。

本项目不仅使用了数据集自带的 561 维特征，还实现了一套**从原始惯性信号 (Inertial Signals) 出发**的特征提取流程，并对比了 Logistic Regression、SVM 和 Random Forest 三种模型的效果。

---

## 📂 项目结构

```text
har-ml-lab/
├── data/
│   ├── loader.py       # 数据搬运工：读取硬盘上的 TXT 文件
│   ├── preprocess.py   # 预处理流水线：切分验证集、标准化
│   ├── features.py     # 特征工程：从原始波形算均值、方差等指标
│   └── __init__.py     # 常量配置
├── models/
│   └── classic.py      # 模型库：封装 LR, SVM, RF
├── reports/            # (自动生成) 存放实验报告和图表
├── tests/              # 单元测试
├── plots.py            # 绘图工具：混淆矩阵、对比图
├── report.py           # 报告生成器：Markdown 导出
├── main.py             # 总指挥：CLI 命令行入口
└── requirements.txt    # 依赖包列表
```

---

## 🛠️ 安装与环境

1.  **创建虚拟环境**:
    ```bash
    cd har-ml-lab
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **数据准备 (Data Setup)**:
    *   **自动下载 (推荐)**: 直接运行 `python main.py`，程序会自动检测并下载 UCI HAR 数据集。
    *   **手动下载**: 下载 [UCI HAR Dataset.zip](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip) 并解压到 `data/raw/UCI HAR Dataset/`。

---

## 🏃‍♂️ 快速开始 (Quick Start)

### 1. 跑 Baseline (使用官方 561 维特征)
这是最简单的模式，直接用逻辑回归跑官方特征：
```bash
python main.py --model lr
```
*预期准确率: ~96%*

### 2. 跑进阶模型 (SVM / Random Forest)
```bash
python main.py --model svm --C 10
python main.py --model rf --rf-trees 200
```
*预期准确率: ~98%*

### 3. 跑自定义特征 (Stage 3 挑战任务) 🔥
不使用官方特征，而是从原始波形自己算特征（63 维）：
```bash
python main.py --use-custom-features --model rf
```
*预期准确率: ~97.8% (惊人的性价比！)*

### 4. 生成报告与图表 📊
加上 `--save-plots` 参数，程序会在 `reports/` 目录下生成混淆矩阵图和 Markdown 实验报告：
```bash
python main.py --use-custom-features --model rf --save-plots
```

---

## 🔬 实验结果概览

| 模型 (Model) | 特征 (Features) | 维度 | 准确率 (Accuracy) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Official | 561 | 96.1% | Baseline |
| **SVM (RBF)** | Official | 561 | 98.2% | 最佳性能 |
| **Random Forest** | Official | 561 | 97.5% | 稳健 |
| **Random Forest** | **Custom** | **63** | **97.8%** | **高光时刻: 仅用 1/9 特征维度** |

---

## 🧪 运行测试

本项目包含自动化测试，确保数据读取和特征计算逻辑正确：
```bash
pytest tests/
```

---

## 📚 References

1. **UCI HAR Dataset**: Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
   [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

---

*Project by [https://github.com/18244241528jm-cpu](https://github.com/18244241528jm-cpu), 2025*

