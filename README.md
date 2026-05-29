<div align="center">

# kaggle-multilingual-nli

**다국어 자연어 추론 (NLI) — Kaggle "Contradictory, My Dear Watson" 솔루션**
**Multilingual NLI — Kaggle "Contradictory, My Dear Watson" solution**

![Language](https://img.shields.io/badge/language-Python%203.10-3776AB?logo=python&logoColor=white)
[![Verify](https://github.com/jumincho/kaggle-multilingual-nli/actions/workflows/verify.yml/badge.svg)](https://github.com/jumincho/kaggle-multilingual-nli/actions/workflows/verify.yml)
![Framework](https://img.shields.io/badge/framework-PyTorch%20%2B%20Transformers-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/model-XLM--RoBERTa-FFD43B?logo=huggingface&logoColor=black)
![Val Acc](https://img.shields.io/badge/val%20acc-0.833-green)
![License](https://img.shields.io/badge/license-MIT-green)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

---

## 개요

> 다국어 자연어 추론(NLI) — Kaggle 경진대회 솔루션

15개 언어의 전제(premise)와 가설(hypothesis) 문장 쌍을
**수반(entailment) / 중립(neutral) / 모순(contradiction)** 세 클래스로 분류하는
[Kaggle 경진대회](https://www.kaggle.com/competitions/contradictory-my-dear-watson)
솔루션입니다. 영어뿐 아니라 아랍어·중국어·힌디어·스와힐리어 등이 섞여 있어
다국어 사전학습 백본이 필요합니다.

<a target="_blank" href="https://colab.research.google.com/github/jumincho/kaggle-multilingual-nli/blob/main/notebooks/contradictory_my_dear_watson.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 접근 방법

- **백본**: [`symanto/xlm-roberta-base-snli-mnli-anli-xnli`](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli) — SNLI/MNLI/ANLI/XNLI에서 사전 파인튜닝된 XLM-RoBERTa
- **분류 헤드**: `[CLS]` 풀링 → Dropout → Linear(768→512) → LayerNorm → ReLU → Dropout → Linear(512→3)
- **전처리**: `premise` + `hypothesis`를 한 시퀀스로 페어 토큰화 (`truncation=True`). 학습 데이터는 75/25 비율로 train/validation 분할
- **학습**: Hugging Face `Trainer`, `AdamW` (`optim="adamw_torch"`), 5 epoch, epoch 단위 평가
- **평가 지표**: accuracy, micro-F1 (`evaluate` 라이브러리)

## 결과

5 에폭 학습 후 validation 기준:

| Epoch | Train Loss | Val Loss | Accuracy |  F1   |
| :---: | :--------: | :------: | :------: | :---: |
|   1   |   0.6683   |  0.5721  |  0.7693  | 0.769 |
|   2   |   0.4966   |  0.7378  |  0.7868  | 0.787 |
|   3   |   0.3754   |  0.8749  |  0.8152  | 0.815 |
|   4   |   0.2022   |  0.9899  |  0.8264  | 0.826 |
|   5   |   0.1202   |  1.1052  |  0.8327  | 0.833 |

최종 validation **accuracy 0.8327**, **F1 0.833**.

## 기술 스택

- **언어**: Python 3.10+
- **딥러닝**: PyTorch ≥ 2.0
- **NLP**: Hugging Face `transformers` (≥ 4.41, < 4.46) · `datasets` · `evaluate` · `accelerate`
- **데이터**: `pandas`, `scikit-learn`
- **시각화**: `matplotlib`, `seaborn`, `plotly`

(버전 범위는 `requirements.txt` 참고 — `transformers`의 `AdamW` 제거, `eval_strategy` 인자명 변경 등 호환 이슈에 맞춰 고정했습니다.)

## 프로젝트 구조

```
kaggle-multilingual-nli/
├── src/
│   ├── __init__.py
│   ├── data.py            # CSV 로드, train/val 분할, 토큰화
│   ├── model.py           # XLMRobertaNLIClassifier + HeadConfig
│   └── train.py           # 학습+예측 파이프라인 (노트북·CLI 공용)
├── notebooks/
│   └── contradictory_my_dear_watson.ipynb   # src/ 모듈을 사용하는 진입 노트북
├── data/
│   ├── train.csv          # id, premise, hypothesis, lang_abv, language, label
│   ├── test.csv           # id, premise, hypothesis, lang_abv, language
│   └── sample_submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## 실행 방법

### Google Colab

상단의 **Open In Colab** 배지로 GPU 환경에서 바로 실행합니다. 노트북은
저장소의 `src/` 모듈을 임포트하므로, Colab에서는 저장소를 clone한 뒤
`notebooks/` 안의 파일을 여는 방식을 권장합니다.

### 로컬

```bash
git clone https://github.com/jumincho/kaggle-multilingual-nli.git
cd kaggle-multilingual-nli
pip install -r requirements.txt

# 노트북 진입
jupyter lab notebooks/contradictory_my_dear_watson.ipynb

# 또는 CLI 한 줄로 학습+예측+제출 파일 생성
python -m src.train --epochs 5 --batch-size 16 --seed 42
```

GPU(CUDA) 권장. 학습 시간은 T4 기준 약 25분, 결과는 `submission.csv`로 저장됩니다.

## 라이선스

[MIT License](./LICENSE)

---

<a name="english"></a>

## English

> Multilingual NLI — solution for the Kaggle "Contradictory, My Dear Watson" competition.

Classify premise–hypothesis pairs in 15 languages as
**entailment / neutral / contradiction**. The data mixes English with
Arabic, Chinese, Hindi, Swahili and other low-resource languages, so a
multilingual pretrained backbone is required.

<a target="_blank" href="https://colab.research.google.com/github/jumincho/kaggle-multilingual-nli/blob/main/notebooks/contradictory_my_dear_watson.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Approach

- **Backbone**: [`symanto/xlm-roberta-base-snli-mnli-anli-xnli`](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli) — XLM-RoBERTa pre-finetuned on SNLI/MNLI/ANLI/XNLI.
- **Head**: `[CLS]` pooling → Dropout → Linear(768→512) → LayerNorm → ReLU → Dropout → Linear(512→3).
- **Preprocessing**: pair-encode `premise` and `hypothesis` (`truncation=True`); 75/25 train/validation split.
- **Training**: Hugging Face `Trainer`, `AdamW` (`optim="adamw_torch"`), 5 epochs, per-epoch evaluation.
- **Metrics**: accuracy and micro-F1 via `evaluate`.

### Results

After 5 epochs (validation):

| Epoch | Train Loss | Val Loss | Accuracy |  F1   |
| :---: | :--------: | :------: | :------: | :---: |
|   1   |   0.6683   |  0.5721  |  0.7693  | 0.769 |
|   2   |   0.4966   |  0.7378  |  0.7868  | 0.787 |
|   3   |   0.3754   |  0.8749  |  0.8152  | 0.815 |
|   4   |   0.2022   |  0.9899  |  0.8264  | 0.826 |
|   5   |   0.1202   |  1.1052  |  0.8327  | 0.833 |

Final validation **accuracy 0.8327**, **F1 0.833**.

### Stack

- **Language**: Python 3.10+
- **DL**: PyTorch ≥ 2.0
- **NLP**: Hugging Face `transformers` (≥ 4.41, < 4.46), `datasets`, `evaluate`, `accelerate`
- **Data**: `pandas`, `scikit-learn`
- **Viz**: `matplotlib`, `seaborn`, `plotly`

(Pinned in `requirements.txt` against breaking changes — `AdamW` removal, `eval_strategy` rename, etc.)

### Layout

```
kaggle-multilingual-nli/
├── src/
│   ├── __init__.py
│   ├── data.py            # CSV load, train/val split, tokenization
│   ├── model.py           # XLMRobertaNLIClassifier + HeadConfig
│   └── train.py           # train + predict pipeline (notebook & CLI)
├── notebooks/
│   └── contradictory_my_dear_watson.ipynb   # thin entry over src/
├── data/
│   ├── train.csv          # id, premise, hypothesis, lang_abv, language, label
│   ├── test.csv           # id, premise, hypothesis, lang_abv, language
│   └── sample_submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

### Run

#### Google Colab

Use the **Open In Colab** badge above. The notebook imports from `src/`, so
the recommended Colab flow is `git clone` the repo and open the notebook
from inside `notebooks/`.

#### Local

```bash
git clone https://github.com/jumincho/kaggle-multilingual-nli.git
cd kaggle-multilingual-nli
pip install -r requirements.txt

# Notebook entry
jupyter lab notebooks/contradictory_my_dear_watson.ipynb

# Or CLI: train + predict + write submission.csv
python -m src.train --epochs 5 --batch-size 16 --seed 42
```

GPU (CUDA) recommended. ~25 minutes on a Colab T4. Output: `submission.csv`.

### License

[MIT License](./LICENSE)
