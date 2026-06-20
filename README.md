# 🧾 Optical Character Recognition - Receipt Text Detection

목적 : 일반인이 다양한 환경에서 촬영한 영수증 이미지에서 텍스트 영역의 위치를 정확하게 검출하는 Text Detection 모델 구현

---

## 📂 ReadME Index
[🎯 Project Overview (프로젝트 개요 및 목표)](#project-overview)

[⏱️ Project Duration & 🔧 Tech Stack (기간 및 기술스택)](#projectduration-techstack)

[📊 Data Analysis & Hypothesis (데이터 분석 및 실험 방향성 설정)](#data-analysis)

[🚀 Experimental Progression (실험 과정 및 빌드업)](#experimental-progression)

[🧪 Final SOTA & Experiment Results (핵심 실험 결과 전체)](#final-sota)

[🛠️ Troubleshooting & Engineering (문제 해결 및 인프라 안정화)](#troubleshooting-engineering)

[📈 Retrospective & Future Work (회고 및 향후 계획)](#retrospective-futurework)

---

<a id="project-overview"></a>

## 🎯 Project Overview

### 프로젝트 배경
영수증에는 구매 일시 · 장소 · 품목 · 금액 등 다양한 정보가 담겨 있어 재무 관리, 소비 패턴 분석, 자동 회계 처리 등에 활용될 수 있지만, 현재 OCR 기술로도 인식이 어려운 글자 패턴이 많아 여전히 수작업에 의존하는 비중이 높습니다. 이를 해결하기 위한 **OCR Pipeline(전처리 → 검출 → 인식)** 중 가장 근본적인 단계인 **글자 검출(Text Detection)**의 강건성을 높이는 모델을 연구합니다.

### 핵심 과제
3,273장의 학습 이미지와 404장의 검증 이미지로 모델을 학습하고, 413장의 평가 이미지에 대해 텍스트 영역의 **Polygon 좌표**를 정확하게 예측하는 것이 목표입니다. 영수증마다 길이 · 폰트 · 두께 · 레이아웃이 제각각이고, 촬영 각도에 따른 왜곡 · 복잡한 배경 · 바코드 포함 등 학습 데이터에 없던 Unseen case가 등장할 가능성이 높아, **Generalization**과 **Optimization** 사이의 최적점을 찾는 것이 핵심 과제입니다.

### 핵심 평가 지표

평가는 [CLEval](https://github.com/clovaai/CLEval) (Character-Level Evaluation) 기반 **H-Mean**으로 이루어집니다. 일반적인 IoU / DetEval 방식은 하나의 단어가 여러 박스로 분리 검출되는 경우에도 강하게 감점되지만, CLEval은 문자(Character) 단위로 평가해 이런 분리 검출에 대한 불합리한 패널티를 줄입니다.

```
예) 정답 단어 : "RIVERSIDE"
   예측 A : "RIVERSIDE"        (하나의 박스로 정확히 검출)
   예측 B : "RIVER" + "SIDE"   (두 개의 박스로 분리 검출)

   → IoU / DetEval : 예측 B는 분리 검출로 간주되어 감점
   → CLEval        : Character 단위로는 누락 · 초과가 없으므로 두 예측 모두 정상 평가
```

- **Detection 전용 평가** : Recognition/Transcription 정보는 사용하지 않음
- **POLY 방식** : GT가 Polygon 기준으로 라벨링되어 있어 QUAD가 아닌 POLY 방식으로 평가
- **제약** : 이미지 1장당 **최대 글자 영역 500개**까지만 평가 대상(초과분 제외), 4점 미만 Polygon은 무시
- **순위 산정** : Public / Private **50:50** 랜덤 split, H-Mean은 소수점 4자리까지 계산(동점 시 선제출 우선)

---

<a id="projectduration-techstack"></a>

## ⏱️ Project Duration & 🔧 Tech Stack

### ⏱️ Project Duration
- 05.06 ~ 05.14 (제출 기록 기준, 총 9일)

### 🔧 Tech Stack
| Category | Tech Stack |
| :--- | :--- |
| **Language** | Python |
| **Framework** | PyTorch, PyTorch Lightning |
| **Config Management** | Hydra (`preset` 기반 YAML 조합) |
| **Encoder (Backbone)** | EfficientNet-B0 (timm, pretrained), ConvNeXt-Small / ConvNeXt-Tiny |
| **Decoder / Head** | UNet (FPN 계열), ASF(Adaptive Scale Fusion, DBNet++), DBHead (Differentiable Binarization) |
| **Augmentation** | Albumentations (HueSaturationValue, GaussianBlur, HorizontalFlip 등) |
| **Evaluation** | CLEval (Character-Level Evaluation) |
| **Environment** | V100 GPU Server (대회 제공 환경) |

---

<a id="data-analysis"></a>

## 📊 Data Analysis & Hypothesis

모델 구조를 바꾸기 전에, 먼저 데이터 자체의 특성과 베이스라인의 약점을 파악해 실험 방향성을 수립했습니다.

> ※ 아래 이미지 경로는 예시이며, 첨부하신 `ocr대회 데이터셋 탐색.pdf` / `EDA.pdf`에서 추출한 실제 차트로 교체해 사용하시면 됩니다.

### Insight 1. 이미지 해상도와 텍스트 크기의 관계

<p align="Left">
  <img src="assets/EDA_height_distribution.png" width="800">
</p>

- **분석** : 학습 이미지의 높이 분포를 확인한 결과 **1279px 근처에 다수의 이미지가 집중**되어 있었습니다. 베이스라인의 640 해상도는 원본 크기에 비해 상당히 작아, 작은 글자가 리사이즈 과정에서 뭉개질 가능성이 높았습니다.
- **실험 방향** : 이미지 사이즈를 1024로 올려 원본 해상도를 더 폭넓게 활용하고, 더 강한 표현력을 가진 인코더(ConvNeXt 계열)도 함께 비교 실험하기로 결정했습니다.

### Insight 2. Precision은 높은데 Recall이 낮다 → 후처리 기준이 너무 엄격하다

- **분석** : 베이스라인 제출 결과 Precision(0.9651)은 높지만 Recall(0.8194)은 상대적으로 낮았습니다. GT와 예측 박스를 시각화해 비교한 결과, 예측한 박스 자체는 정확했지만 작은 글자나 점선 영역에서 검출을 놓치는 경우가 많았습니다.
- **실험 방향** : 모델 구조를 바꾸기 전에 먼저 DBPostProcessor의 `thresh`(이진화 기준)와 `box_thresh`(박스 확정 기준)를 낮춰 더 많은 영역을 검출 대상으로 포함시키는 실험을 우선순위에 두었습니다.

### Insight 3. val / train 데이터 분포가 유사하다 → 동일 전처리 전략 적용 가능

- **분석** : val 셋과 train 셋의 이미지 분포 및 평균값을 비교한 결과 두 데이터셋의 양상이 유사했습니다.
- **실험 방향** : 별도의 val 전용 전처리 파이프라인을 구성할 필요 없이, train과 동일한 리사이즈 · 정규화 전략을 val · test에도 그대로 적용하기로 결정했습니다.

### Insight 4. 영수증 데이터 특유의 노이즈 패턴 → 강건한 증강 · 후처리 필요

<p align="Left">
  <img src="assets/EDA_noise_samples.png" width="800">
</p>

- **분석** : 이미지를 직접 살펴본 결과 워터마크, 구겨진 영수증, 볼펜 자국, 도장, 이물질 번짐이 GT에 포함된 경우와 영수증 뒷면 비침이 GT로 검출된 경우, 바코드 하단 숫자의 검출 방식 불일치(하나의 박스 vs 개별 박스) 등 다양한 노이즈 케이스를 확인했습니다. 또한 카페 영수증은 짧고 박스 수가 적은 반면, 마트 영수증은 길고 박스 수가 많아 영수증 종류에 따른 편차도 컸습니다.
- **실험 방향** : 이런 촬영 환경 · 노이즈 편차에 대응하기 위해 색상 · 채도 · 블러 계열의 증강을 우선 적용하기로 했고, 점선(`---`)처럼 평가 대상에 포함되는 GT는 임의로 제거하지 않도록 주의했습니다.

---

<a id="experimental-progression"></a>

## 🚀 Experimental Progression

총 4단계의 점진적 실험을 통해 H-Mean **0.8818 → 0.9853**으로 성능을 끌어올렸습니다.

### Phase 1. 베이스라인 구축 및 문제 진단

- **베이스라인 실행 및 EDA** : 기본 DBNet 베이스라인을 10 epoch 학습해 첫 제출을 완료했습니다(H-Mean 0.8818, Precision 0.9651, Recall 0.8194). Precision은 높지만 Recall이 낮아 검출 자체를 놓치는 박스가 많다는 점을 확인했고, GT와 예측 결과를 시각화해 비교한 결과 작은 글자와 점선 영역의 누락이 두드러졌습니다. 또한 EDA를 통해 이미지 높이가 1279px 근처에 집중된 분포를 확인해, 이후 해상도 실험의 근거를 마련했습니다.

### Phase 2. 해상도 · 인코더 · 옵티마이저로 기반 다지기

- **해상도 1024 + 인코더 비교** : 이미지 사이즈를 640 → 1024로 올리고, EfficientNet-B0와 ConvNeXt-Small 인코더를 비교했습니다(H-Mean 0.9454 / 0.9466). ConvNeXt가 소폭 우세했지만 학습 시간이 과도해 이후 실험은 EfficientNet-B0를 기준으로 진행했습니다. 이 과정에서 ConvNeXt에 기존 lr(0.001)을 그대로 적용해 H-Mean이 0.000까지 떨어지는 문제를 겪었는데, lr을 1e-4로 낮춰 해결했습니다(자세한 내용은 Troubleshooting 참고). 또한 StepLR(step_size=100)이 20 epoch 내에서 거의 작동하지 않는다는 점을 발견해 CosineAnnealingLR로 교체했습니다.
- **옵티마이저 / 스케줄러 정착** : AdamW(lr=0.0005, weight_decay=0.01) + CosineAnnealingLR(T_max=30, eta_min=1e-6) 조합을 적용했습니다(H-Mean 0.9463). 단독 효과는 크지 않았지만, 이후 모든 실험의 기본 설정으로 유지했습니다.

### Phase 3. 후처리 튜닝과 증강 실험으로 SOTA 달성 ⭐

- **후처리 파라미터 튜닝** : EDA에서 확인한 Recall 저하 문제를 해결하기 위해 `thresh`를 0.3 → 0.2로 낮추고, `max_candidates`를 300 → 500(대회 규정 최대치)으로, `use_polygon`을 False → True(CLEval POLY 방식 대응)로 변경했습니다. 모델 구조를 전혀 바꾸지 않았는데도 H-Mean이 0.9463 → **0.9844(+0.0381)**로 전체 실험 중 가장 큰 폭으로 상승했습니다.
- **데이터 증강 실험** : 처음에는 CLAHE, RandomBrightnessContrast, HueSaturationValue, Blur, H-flip을 한꺼번에 적용했지만 ByteTensor 에러와 학습 중단 문제로 H-Mean이 0.9733까지 하락했습니다(원인은 Troubleshooting 참고). 이를 HueSaturationValue + GaussianBlur + HorizontalFlip 세 가지로 줄이고, GT 박스 손실을 유발하던 ShiftScaleRotate를 제거하자 H-Mean **0.9853**으로 최종 SOTA를 달성했습니다.

### Phase 4. DBNet++ ASF 모듈 고도화 탐색

- **ASF(Adaptive Scale Fusion) 모듈 구현** : DBNet의 단순 합산 방식 Decoder를 학습 가능한 가중합 구조로 개선한 DBNet++의 ASF 모듈을 논문 저자 공식 구현을 참고해 구현했습니다. 그러나 신규 레이어의 랜덤 초기화로 학습 초반 피처가 불안정해져 epoch=30 기준 H-Mean이 0.9838로 SOTA에 못 미쳤습니다.
- **안정화 시도** : epoch을 30 → 50으로 늘리고, Spatial Attention 마지막 Conv의 bias를 1로 초기화해 Sigmoid 초기 출력을 0.5 → 0.73으로 높여 초반 피처 억제를 완화했습니다(epoch=31, H-Mean 0.9851로 SOTA에 근접). 이후 ConvNeXt-Tiny + DBNet++ 조합도 시도했지만(H-Mean 0.9843), 대회 마감 시간 부족으로 epoch=18에서 조기 종료해 충분한 수렴을 확인하지 못했습니다.

---

<a id="final-sota"></a>

## 🧪 Final SOTA & Experiment Results

### 🏆 Final SOTA 아키텍처

<p align="Left">
  <img src="assets/architecture.svg" width="600">
</p>

> 최종 제출 모델 : EfficientNet-B0 Encoder + UNet Decoder + DBHead + 후처리 튜닝(thresh=0.2, box_thresh=0.4) + HueSaturationValue/GaussianBlur/H-flip 증강

---

### 📊 전체 실험 결과 테이블

| 버전 | H-Mean | Precision | Recall | 핵심 변경 | 결과 | 인사이트 |
|------|--------|-----------|--------|-----------|------|----------|
| baseline | 0.8818 | 0.9651 | 0.8194 | DBNet 베이스라인 (640 해상도) | — | Precision은 높으나 Recall이 낮아 검출 누락 확인 |
| V1_efficientnet_b0 | 0.9454 | 0.9801 | 0.9182 | 해상도 640 → 1024 | ⬆️ | 해상도만 올려도 큰 폭 개선, 작은 글자 검출력 향상 |
| V2_ConvNeXt | 0.9466 | 0.9813 | 0.9184 | 인코더 EfficientNet-B0 → ConvNeXt-Small | ⬆️ (소폭) | lr=0.001에서 H-Mean 0.000 발생(Catastrophic Forgetting), lr=1e-4로 해결했으나 학습 시간 과다로 이후 미채택 |
| V1-1_AdamW_CosineLR | 0.9463 | 0.9803 | 0.9188 | AdamW + CosineAnnealingLR 적용 | ➡️ | 단독 효과는 미미하나 이후 모든 실험의 기본 설정으로 유지 |
| **V1-2_postprocess_tuning** | **0.9844** | **0.9844** | **0.9849** | **thresh 0.3→0.2, max_candidates 300→500, use_polygon True** | **⬆️⬆️ 단일 최대 향상** | 모델 구조 변경 없이 후처리 설정만으로 +0.0381 달성 |
| V1-4_aug_aggressive | 0.9733 | 0.9795 | 0.9711 | CLAHE 등 5종 증강 동시 적용 | ⬇️ | 과도한 증강 조합은 오히려 성능 저하 (ByteTensor 에러, lr 수렴 이슈 동반) |
| **V1-4_aug_basic (최종 제출)** | **0.9853** | 0.9843 | 0.9867 | **HueSaturationValue + GaussianBlur + H-flip만 적용** | **⬆️ 최종 SOTA** | 기하변환(ShiftScaleRotate) 제거로 GT 손실 방지, 최소 증강 조합이 최고 성능 |
| V3_DBNetpp_ASF (epoch=30) | 0.9838 | 0.9842 | 0.9841 | ASF(Adaptive Scale Fusion) 모듈 추가 | ⬇️ (SOTA 대비) | 신규 레이어 랜덤 초기화로 학습 초반 불안정 |
| V3-1_DBNetpp_bias_init (epoch=31) | 0.9851 | 0.9853 | 0.9854 | Spatial Attention bias=1 초기화 | ⬆️ (V3 대비) | Sigmoid 초기 출력 0.5→0.73로 피처 억제 완화, SOTA 근접 |
| V4_ConvNeXt-Tiny_DBNetpp (epoch=18) | 0.9843 | 0.9845 | 0.9846 | 인코더 + ASF 동시 적용 | ➡️ | 대회 마감으로 조기 종료, 추가 학습 시 개선 여지 |

---

<a id="troubleshooting-engineering"></a>

## 🛠️ Troubleshooting & Engineering

### 1. ConvNeXt 학습 붕괴 (lr 설정 오류로 H-Mean 0.000)

#### 문제 정의
인코더를 EfficientNet-B0에서 ConvNeXt-Small로 교체하고 기존 lr(0.001)을 그대로 적용했더니, 학습이 정상적으로 진행되지 않아 H-Mean이 **0.000**까지 떨어졌습니다.

#### 원인 분석
ConvNeXt처럼 대형 pretrained 모델에 비교적 가벼운 EfficientNet-B0 기준으로 튜닝된 높은 lr을 그대로 적용하면, 기존에 학습된 pretrained weight가 초반 몇 step 만에 손상되는 **Catastrophic Forgetting**이 발생합니다.

#### 해결 방안
lr을 0.001 → 1e-4로 낮춰 재학습했습니다. epoch 2부터 H-Mean이 0.969까지 빠르게 회복되었고, 최종적으로 0.9466을 기록했습니다.

```yaml
# ❌ Before: EfficientNet 기준 lr을 ConvNeXt에 그대로 적용 (H-Mean 0.000)
optimizer:
  lr: 0.001

# ✅ After: 대형 pretrained 모델에 맞춰 lr 하향 조정
optimizer:
  lr: 0.0001   # 1e-4, epoch 2부터 H-Mean 0.969로 회복
```

#### 인사이트
모델 크기와 pretrained 정도에 따라 lr을 비례적으로 조정해야 합니다. 작은 모델에서 튜닝된 하이퍼파라미터를 더 큰 모델에 그대로 이식하면 위험할 수 있으므로, 인코더를 교체할 때는 lr도 함께 재탐색하는 것이 안전합니다.

### 2. 공격적 데이터 증강 조합에서의 학습 실패 (복합 원인)

#### 문제 정의
일반화 성능 향상을 위해 CLAHE + RandomBrightnessContrast + HueSaturationValue + Blur + H-flip을 동시에 적용했는데, 학습 중 **ByteTensor 에러**가 발생했고, 이를 해결한 뒤에도 H-Mean이 0.9733으로 당시 SOTA(0.9844) 대비 하락했습니다.

#### 원인 분석
1. **타입 충돌** : GaussNoise, ImageCompression의 출력이 uint8 타입으로 유지되어, 이후 텐서 변환 과정에서 타입 충돌(ByteTensor 에러)이 발생했습니다.
2. **스케줄러 수렴** : CosineAnnealingLR을 T_max=30으로 설정한 상태에서 30 epoch을 초과해 학습을 지속하자 lr이 eta_min(1e-6)에 수렴해 학습이 사실상 중단되었고, epoch 29에서 H-Mean이 급락했습니다.

#### 해결 방안
GaussNoise, ImageCompression 두 증강을 제거해 타입 에러를 해결하고, 증강 조합을 HueSaturationValue + GaussianBlur + HorizontalFlip으로 단순화했습니다.

```python
# ❌ Before: uint8 유지 증강이 텐서 변환 충돌 유발 + 증강 과다
transform = [CLAHE(), RandomBrightnessContrast(), GaussNoise(),
             ImageCompression(), HueSaturationValue(), Blur(), HorizontalFlip()]
# → ByteTensor 에러, H-Mean 0.9733 (SOTA 대비 하락)

# ✅ After: 타입 충돌 증강 제거 + 최소 조합
transform = [HueSaturationValue(), GaussianBlur(), HorizontalFlip()]
# → 학습 안정화, H-Mean 0.9853 (최종 SOTA)
```

#### 인사이트
증강 기법을 다다익선으로 쌓기보다, 각 기법이 출력 텐서 타입과 학습 스케줄에 미치는 영향까지 점검해야 합니다. "더 많은 증강 = 더 좋은 일반화"라는 직관이 항상 맞지는 않으며, 데이터 특성에 맞는 최소 조합이 오히려 최고 성능을 냈습니다.

### 3. 기하학적 증강(ShiftScaleRotate)으로 인한 Ground Truth 박스 소실

#### 문제 정의
일반화를 높이기 위해 ShiftScaleRotate(회전 · 이동 · 스케일 변환)를 증강에 포함했으나, 일부 이미지에서 **NaN이 발생**하는 현상이 확인되었습니다.

#### 원인 분석
회전 · 이동 변환 시 GT Polygon의 keypoint 일부가 이미지 경계 밖으로 나가는 경우가 발생하는데, `remove_invisible: True` 설정으로 인해 해당 GT 박스 자체가 소실되었습니다. 시각화 결과 손실 비율은 0~2% 수준으로 크지 않았지만, 일부 이미지에서 이 현상이 NaN 발생의 원인으로 추정됩니다.

#### 해결 방안
ShiftScaleRotate를 증강 목록에서 제거하고, 색상 · 채도 · 블러처럼 좌표에 영향을 주지 않는 증강만 유지했습니다. 결과적으로 H-Mean 0.9853으로 최종 SOTA를 달성했습니다.

#### 인사이트
회전 · 이동 같은 기하학적 증강은 Detection Task에서 GT 좌표 보존 설정과 함께 반드시 검증해야 합니다. "더 다양한 변형 = 더 강건한 모델"이라는 가설이 항상 성립하지는 않으며, 좌표 정밀도가 중요한 Text Detection Task에서는 기하 변환의 부작용을 면밀히 살펴야 합니다.

### 4. 기타 환경 설정 이슈

| 문제 | 원인 | 해결 |
|------|------|------|
| **StepLR이 20 epoch 내 사실상 비작동** | `step_size=100`으로 설정되어 의미 있는 decay가 거의 발생하지 않음 | `CosineAnnealingLR`(T_max=30, eta_min=1e-6)로 교체 |
| **DBNet++ ASF 모듈 학습 초반 불안정** | 신규 Spatial Attention Conv 레이어의 랜덤 초기화로 초반 피처가 억제됨 | 마지막 Conv의 bias를 1로 초기화(Sigmoid 출력 0.5→0.73)하고 epoch을 30→50으로 증가 |
| **max_candidates 1000 실험이 무의미** | 대회 규정상 이미지당 500개 초과 영역은 평가 제외 | 500으로 고정 (1000과 점수 동일함을 확인) |

---

<a id="retrospective-futurework"></a>

## 📈 Retrospective & Future Work

### 📌 회고
이미지 데이터셋은 분석과 시각화만으로는 데이터 형태를 직관적으로 파악하기 어렵다는 것을 이번 대회를 통해 깨달았습니다. 이전 문서 분류 대회에서는 증강 전후 이미지만 랜덤하게 비교했을 뿐, GT가 실제로 어떻게 나왔는지, 라벨링에 오류가 없는지까지는 충분히 들여다보지 못했는데, 이번에는 가능한 한 많은 이미지를 직접 확인하고 시각화하려고 노력했습니다. 멘토님께서도 BBox를 시각화하고, 증강 기법의 range나 파라미터 값에 따라 이미지가 실제로 어떻게 달라지는지 직접 보면서 데이터에 맞게 증강을 설계하는 것이 중요하다고 조언해주셔서, 이를 실험 전반에 적용하려고 노력했습니다.

### 📌 아쉬운 점
- DBNet++의 ASF 모듈이 이론적으로는 DBNet보다 성능이 좋다고 알려져 있어 도입했지만, 실제 구현 결과는 오히려 성능이 하락했습니다. 원인 파악과 코드 구조 분석이 쉽지 않아 충분히 디버깅하고 고도화하지 못한 점이 아쉬웠습니다.
- 두 개의 대회를 병행하면서 학습 시간이 짧은 가벼운 모델 위주로 실험을 진행하다 보니, 더 큰 backbone이나 더 많은 epoch으로 끝까지 밀어붙여 보지 못한 점도 아쉬움으로 남았습니다.

### 📗 향후 계획
- DBNet++ ASF 모듈이 기대만큼 성능을 내지 못한 원인을 코드 레벨에서 더 깊게 분석하고, bias 초기화 외에 다른 안정화 기법(Warm-up, Layer-wise lr 등)도 적용해 재실험해보고 싶습니다.
- 대회 일정 제약 없이 충분한 시간을 확보한다면, ConvNeXt 계열 등 더 큰 backbone과 더 많은 epoch으로 DBNet++ 구조의 실제 성능을 검증해보고 싶습니다.