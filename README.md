# Contradictory, My Dear Watson

> 다국어 자연어 추론(NLI) — Kaggle 경진대회 솔루션

15개 언어로 된 전제(premise)와 가설(hypothesis) 문장 쌍을
**수반(entailment) / 중립(neutral) / 모순(contradiction)** 세 가지로 분류하는
[Kaggle 경진대회](https://www.kaggle.com/competitions/contradictory-my-dear-watson)
참가 솔루션입니다. 영어뿐 아니라 아랍어, 중국어, 힌디어, 스와힐리어 등
다양한 언어가 섞여 있어 다국어 사전학습 모델이 필수입니다.

<a target="_blank" href="https://colab.research.google.com/github/jumincho/contradictory-my-dear-watson/blob/main/notebooks/contradictory_my_dear_watson.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 접근 방법

- **모델**: [`symanto/xlm-roberta-base-snli-mnli-anli-xnli`](https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli)
  — SNLI / MNLI / ANLI / XNLI 로 사전 파인튜닝된 XLM-RoBERTa 기반 모델을 백본으로 사용
- **커스텀 헤드**: 768 → 512 (LayerNorm + ReLU + Dropout) → 3-class 분류기
- **전처리**: `premise`, `hypothesis` 쌍을 하나의 시퀀스로 토큰화 (`truncation=True`).
  학습 데이터는 75 / 25 비율로 train / validation 분할
- **학습**: Hugging Face `Trainer`, `AdamW`, 5 epoch, `evaluation_strategy="epoch"`
- **평가 지표**: accuracy, F1 (`evaluate` 라이브러리)

## 결과

5 에폭 학습 후 validation 기준:

| Epoch | Training Loss | Validation Loss | Accuracy |  F1   |
| :---: | :-----------: | :-------------: | :------: | :---: |
|   1   |    0.6683     |     0.5721      |  0.7693  | 0.769 |
|   2   |    0.4966     |     0.7378      |  0.7868  | 0.787 |
|   3   |    0.3754     |     0.8749      |  0.8152  | 0.815 |
|   4   |    0.2022     |     0.9899      |  0.8264  | 0.826 |
|   5   |    0.1202     |     1.1052      |  0.8327  | 0.833 |

최종 validation accuracy **0.8327**, F1 **0.833**.

## 기술 스택

- **언어**: Python 3
- **딥러닝**: PyTorch, TensorFlow (GPU 환경 설정용)
- **NLP**: Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`
- **데이터**: pandas, scikit-learn (`train_test_split`)
- **시각화**: matplotlib, seaborn, plotly
- **실험 관리**: Weights & Biases (선택)

## 프로젝트 구조

```
contradictory-my-dear-watson/
├── notebooks/
│   └── contradictory_my_dear_watson.ipynb   # 전체 파이프라인 (전처리 → 학습 → 제출)
├── data/
│   ├── train.csv                            # 학습 데이터 (premise, hypothesis, label, language)
│   ├── test.csv                             # 평가 데이터
│   └── sample_submission.csv                # 제출 양식
├── requirements.txt
└── README.md
```

## 실행 방법

### Google Colab

상단의 "Open In Colab" 배지를 클릭하면 GPU 환경에서 바로 실행할 수 있습니다.
노트북 내 데이터 경로는 `../data/*.csv` 로 설정되어 있으므로, Colab에서는
`data/` 디렉터리를 통째로 업로드하거나 저장소를 clone 한 뒤 `notebooks/`
디렉터리로 이동해 실행하세요.

### 로컬

```bash
git clone https://github.com/jumincho/contradictory-my-dear-watson.git
cd contradictory-my-dear-watson
pip install -r requirements.txt
jupyter notebook notebooks/contradictory_my_dear_watson.ipynb
```

GPU(CUDA) 환경을 권장합니다. 학습 시간은 T4 기준 약 25분.

## 스크린샷

![004](https://github.com/jumincho/Contradictory-My-Dear-Watson/assets/77545063/e79d7714-4524-4840-9c42-4805504be058)
![005](https://github.com/jumincho/Contradictory-My-Dear-Watson/assets/77545063/a38c572e-029e-407c-9861-b6844bf82e91)
![006](https://github.com/jumincho/Contradictory-My-Dear-Watson/assets/77545063/d891de22-b999-4108-9628-f8743492e19b)
