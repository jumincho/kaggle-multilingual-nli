<div align="center">

# kaggle-multilingual-nli

**多语言自然语言推理 (NLI) — Kaggle "Contradictory, My Dear Watson" 解决方案**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![Val Acc](https://img.shields.io/badge/val%20acc-0.833-green)
![License](https://img.shields.io/badge/license-MIT-green)

[한국어](./README.md) · [English](./README.md#english) · **中文**

</div>

---

## 概览

> 多语言自然语言推理 (NLI) — Kaggle 竞赛解决方案

将 15 种语言的「前提 (premise) — 假设 (hypothesis)」句对分类为
**蕴含 (entailment) / 中立 (neutral) / 矛盾 (contradiction)** 三类的
[Kaggle 比赛](https://www.kaggle.com/competitions/contradictory-my-dear-watson)
解决方案。数据中除英语外还混合了阿拉伯语、中文、印地语、斯瓦希里语等低资源语言,
因此必须使用多语言预训练骨干。

<a target="_blank" href="https://colab.research.google.com/github/jumincho/kaggle-multilingual-nli/blob/main/notebooks/contradictory_my_dear_watson.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 方法

- **骨干模型**: [`symanto/xlm-roberta-base-snli-mnli-anli-xnli`](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli) — 已在 SNLI/MNLI/ANLI/XNLI 上预微调的 XLM-RoBERTa。
- **分类头**: `[CLS]` 池化 → Dropout → Linear(768→512) → LayerNorm → ReLU → Dropout → Linear(512→3)。
- **预处理**: 将 `premise` 与 `hypothesis` 作为单一序列成对编码 (`truncation=True`);训练集按 75/25 划分为 train/validation。
- **训练**: Hugging Face `Trainer`、`AdamW` (`optim="adamw_torch"`)、5 个 epoch、每个 epoch 评估一次。
- **指标**: 通过 `evaluate` 库计算 accuracy 与 micro-F1。

## 结果

5 个 epoch 训练后的 validation 指标:

| Epoch | Train Loss | Val Loss | Accuracy |  F1   |
| :---: | :--------: | :------: | :------: | :---: |
|   1   |   0.6683   |  0.5721  |  0.7693  | 0.769 |
|   2   |   0.4966   |  0.7378  |  0.7868  | 0.787 |
|   3   |   0.3754   |  0.8749  |  0.8152  | 0.815 |
|   4   |   0.2022   |  0.9899  |  0.8264  | 0.826 |
|   5   |   0.1202   |  1.1052  |  0.8327  | 0.833 |

最终验证集 **accuracy 0.8327**、**F1 0.833**。

## 技术栈

- **语言**: Python 3.10+
- **深度学习**: PyTorch ≥ 2.0
- **NLP**: Hugging Face `transformers` (≥ 4.36, < 4.46)、`datasets`、`evaluate`、`accelerate`
- **数据**: `pandas`、`scikit-learn`
- **可视化**: `matplotlib`、`seaborn`、`plotly`

(版本范围已在 `requirements.txt` 中锁定 —— 应对 `transformers` 中 `AdamW` 移除、`eval_strategy` 参数改名等不兼容变更。)

## 项目结构

```
kaggle-multilingual-nli/
├── src/
│   ├── __init__.py
│   ├── data.py            # CSV 加载、train/val 划分、分词
│   ├── model.py           # XLMRobertaNLIClassifier + HeadConfig
│   └── train.py           # 训练 + 预测流水线 (notebook 与 CLI 共用)
├── notebooks/
│   └── contradictory_my_dear_watson.ipynb   # 调用 src/ 的入口 notebook
├── data/
│   ├── train.csv          # id, premise, hypothesis, lang_abv, language, label
│   ├── test.csv           # id, premise, hypothesis, lang_abv, language
│   └── sample_submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 运行方式

### Google Colab

点击页面顶部的 **Open In Colab** 徽章即可在 GPU 环境中运行。Notebook 会从
仓库的 `src/` 模块中导入代码,推荐在 Colab 中先 `git clone` 仓库,然后从
`notebooks/` 目录打开 notebook。

### 本地

```bash
git clone https://github.com/jumincho/kaggle-multilingual-nli.git
cd kaggle-multilingual-nli
pip install -r requirements.txt

# Notebook 入口
jupyter lab notebooks/contradictory_my_dear_watson.ipynb

# 或使用 CLI 一行命令完成训练 + 预测 + 生成提交文件
python -m src.train --epochs 5 --batch-size 16 --seed 42
```

推荐使用 GPU (CUDA)。在 T4 上约需 25 分钟,输出文件为 `submission.csv`。

## 许可证

[MIT License](./LICENSE)
