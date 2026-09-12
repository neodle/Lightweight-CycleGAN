<div align="center">

# Lightweight CycleGAN

**CycleGAN 기반 색상화 성능 유지와 처리 속도 향상**

비쌍(unpaired) CycleGAN으로 흑백 → 컬러 변환을, **2배 빠르고 52% 더 가볍게** — 품질 손실 없이.

[English](README.md) · **한국어**

<sub>KAICTS 2025 추계학술발표대회 · 건양대학교</sub>

</div>

---

## 한눈에 보기

기본 CycleGAN 생성기(Residual block 9개, 약 1,137만 파라미터)는 품질은 좋지만 실시간·온디바이스 환경에 쓰기엔 너무 무겁습니다. 본 연구는 생성기를 **Residual block 수**, **채널 폭**, **Depthwise separable convolution** 세 축으로 체계적으로 축소하고, 각 축이 품질에 어떤 대가를 치르는지 측정했습니다.

결과는 직관과 다르면서도 유용합니다. **Residual block을 9개에서 4개로 줄이면 모델이 더 작고, 더 빠르고, 심지어 더 좋아집니다.**

| 지표 | Baseline (9 block) | **경량화 (4 block)** | 변화 |
|---|---:|---:|:--|
| Params | 11.37 M | **5.48 M** | 🔻 52% |
| FLOPs | 113.78 G | **50.90 G** | 🔻 55% |
| FPS | 169.78 | **272.90** | 🔺 1.61배 |
| PSNR ↑ | 20.207 | **21.111** | 🔺 +0.904 |
| SSIM ↑ | 0.816 | **0.852** | 🔺 +0.036 |
| LPIPS ↓ | 0.279 | **0.235** | 🔻 −0.044 |

> 더 작은 모델이 필요하다면, `4 block + 채널 ×0.25` 변형은 **파라미터 0.34M으로 355 FPS** — baseline 대비 33배 작으면서도 PSNR 20.1을 유지합니다.

---

## 목차

1. [연구 배경](#1-연구-배경)
2. [연구 방법](#2-연구-방법)
3. [데이터셋](#3-데이터셋)
4. [학습 설정](#4-학습-설정)
5. [실험 결과](#5-실험-결과)
6. [분석](#6-분석)
7. [한계 및 향후 연구](#7-한계-및-향후-연구)
8. [재현 방법](#8-재현-방법)
9. [인용 및 저자](#9-인용-및-저자)

---

## 1. 연구 배경

**Pix2Pix** 와 같은 지도 학습 기반 모델은 동일 장면의 *쌍(pair)* 데이터에 의존합니다. 그러나 오래된 사진이나 역사 기록물처럼 특수한 상황에서는 쌍 데이터 자체가 존재하지 않거나, 확보에 높은 비용이 요구됩니다.

**CycleGAN** 은 비쌍(unpaired) 데이터로 학습할 수 있어 이 한계를 해결했고, 색상화 과제에 자연스럽게 들어맞습니다. 다만 기존 연구들은 대부분 **출력 품질** 위주로 최적화되어 왔습니다. 기본 생성기는 Residual block 9개를 쌓아 256×256 한 장당 약 114 GFLOPs를 요구하며, 이는 모바일·엣지 디바이스에서의 실시간 활용을 사실상 불가능하게 만듭니다.

**본 연구의 질문은 다릅니다.** 그 생성기 중 실제로 일하고 있는 부분은 얼마나 될까요? 학습 레시피는 그대로 두고 생성기만 압축한 뒤, 품질과 속도를 함께 측정했습니다.

<p align="center">
  <img src="assets/pipeline.png" width="880" alt="실험 파이프라인" />
</p>

<p align="center"><sub>컬러 데이터셋 → 흑백 변환 → 동일 조건에서 baseline과 경량화 생성기 학습 → 각 결과를 Ground Truth와 FID / PSNR / SSIM / LPIPS로 비교 → 성능을 유지하면서 크기·속도가 개선될 때까지 반복</sub></p>

---

## 2. 연구 방법

### 2.1 CycleGAN 구조

CycleGAN은 생성기 2개와 판별기 2개를 동시에 학습합니다. 도메인 A의 이미지를 B로 변환한 뒤 다시 A로 되돌리며, **Cycle Consistency Loss** 가 이 왕복 결과를 원본과 일치시키도록 강제합니다. 덕분에 쌍 데이터 없이도 원본의 구조적 특징이 보존됩니다.

<p align="center">
  <img src="assets/cycle-consistency.jpg" width="760" alt="Cycle consistency: A → B → A′, B → A → B′" />
</p>

<p align="center"><sub>위쪽: real A → fake B → 복원된 A′ / 아래쪽: real B → fake A → 복원된 B′. 학습은 ‖A − A′‖ 와 ‖B − B′‖ 를 최소화합니다.</sub></p>

### 2.2 세 가지 경량화 축

인코더–디코더 골격과 손실 함수는 그대로 두고, 생성기만 변형했습니다.

<p align="center">
  <img src="assets/architecture.png" width="900" alt="Lightweight CycleGAN 생성기 구조 변화" />
</p>

| # | 축 | 변경 내용 |
|---|---|---|
| **A** | *Baseline* | `Conv ×3 (Encoder) → Res-Block ×9 → Transpose Conv ×2 → Conv (Decoder)` |
| **B** | **Residual block 수 축소** | 9 → **6** → **4** |
| **C** | **채널 폭 축소** | 모든 Conv 및 Res-Block 채널을 **×0.75 / ×0.5 / ×0.25** |
| **D** | **Depthwise Separable** | Residual block 내부를 Depthwise Separable Conv + 1×1 bottleneck 으로 재구성 |
| **E** | **결합** | 4 block **+** 채널 폭 축소 (×0.25 / ×0.5 / ×0.75) |

모든 변형은 **동일한 데이터, 해상도, 옵티마이저, 스케줄, 시드**로 처음부터 학습했습니다. 따라서 아래 결과 표의 차이는 오직 구조에서 비롯된 것입니다.

---

## 3. 데이터셋

특정 피사체만 색상화하는 모델이 되지 않도록, 공개 데이터셋 3종을 하나의 도메인 다양성 높은 데이터셋으로 병합했습니다.

| 출처 | 샘플링 | 장수 |
|---|---|---:|
| [Animals Detection Images](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset) | 80개 클래스 × 랜덤 50장 | 약 4,000 |
| [SUN397 (50-50)](https://www.kaggle.com/datasets/lash45/sun397-50-50) — 장소 | 397개 클래스 × 랜덤 10장 | 약 3,970 |
| [StyleGAN-Human](https://stylegan-human.github.io/data.html) — 인물 | 랜덤 추출 | 약 4,000 |
| **합계** | | **약 12,000** |

**전처리**

- 모든 이미지를 **256 × 256** 으로 Resize (모델 입력 크기 고정)
- 각 컬러 이미지를 흑백으로 변환해 평가용 정답 쌍을 구성 (학습 자체는 비쌍 방식 유지)
- **Train 70% / Validation 20% / Test 10%** 로 분할. 아래 모든 수치는 학습에 사용되지 않은 Test 데이터로 측정했습니다.

### 왜 데이터를 병합했는가

동일한 baseline 구조를 동물 데이터만으로, 그리고 병합 데이터로 각각 학습해 비교했습니다.

| 지표 | Animal only | Merged |
|---|---:|---:|
| Params / FLOPs | 동일 | 동일 |
| PSNR ↑ | 19.656 | **20.207** |
| SSIM ↑ | 0.625 | **0.816** |
| LPIPS ↓ | **0.258** | 0.279 |
| Colorfulness ↑ | 19.104 | **29.556** |

Animal only 모델은 중저채도 영역에 머무는 제한적인 색을 냅니다. 반면 병합 모델은 Colorfulness가 19.1 → 29.6으로 크게 오르고, 실제 사람 눈에도 훨씬 다채롭고 풍부한 색감을 보여줍니다. 따라서 **이후 모든 실험의 baseline은 병합 데이터 모델**을 사용했습니다.

---

## 4. 학습 설정

Baseline과 모든 경량화 변형에 동일하게 적용했습니다.

| 항목 | 값 |
|---|---|
| 입력 해상도 | 256 × 256 |
| Batch size | 16 |
| Epochs | 100 (50 고정 + 50 Decay) |
| Optimizer | Adam — `lr = 2e-4`, `β₁ = 0.5`, `β₂ = 0.999` |
| Loss 가중치 | `Cycle = 10.0`, `Identity = 0.5` |
| Seed | 42 |
| Data | Merged (장소 + 동물 + 인물) |

**평가 지표.** 픽셀·구조적 유사도는 PSNR / SSIM, 지각적 거리는 LPIPS, 실시간 처리 속도는 FPS로 평가했습니다. Params와 FLOPs는 생성기 기준입니다.

---

## 5. 실험 결과

### 5.1 전체 비교

| 지표 | Baseline | 6 block | **4 block** | Scale 0.75 | Scale 0.5 | Scale 0.25 | Depthwise | 4b + ×0.25 | 4b + ×0.5 | 4b + ×0.75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Params (M)** | 11.37 | 7.84 | **5.48** | 6.40 | 2.85 | 0.72 | 0.73 | 0.34 | 1.37 | 3.08 |
| **FLOPs (G)** | 113.78 | 70.30 | **50.90** | 31.70 | 6.40 | 0.42 | 12.40 | 3.70 | 3.30 | 16.40 |
| **FPS ↑** | 169.78 | 222.00 | **272.90** | 195.55 | 195.74 | 206.22 | 158.40 | 355.26 | 331.71 | 331.31 |
| **PSNR ↑** | 20.207 | 20.683 | **21.111** | 20.698 | 19.993 | 17.885 | 19.994 | 20.148 | 19.905 | 20.587 |
| **SSIM ↑** | 0.816 | 0.828 | **0.852** | 0.837 | 0.792 | 0.687 | 0.772 | 0.797 | 0.802 | 0.844 |
| **LPIPS ↓** | 0.279 | 0.266 | **0.235** | 0.248 | 0.271 | 0.256 | 0.320 | 0.305 | 0.264 | 0.246 |

세 가지 품질 지표를 모두 최고로 유지하면서 1.6배 빨라진 구성은 **Residual block 4개** 입니다.

### 5.2 정성적 평가

왼쪽 → 오른쪽: **입력(흑백)** · **Baseline CycleGAN** · **경량화 모델 (4 block)** · **Ground Truth**

<p align="center">
  <img src="assets/result-animal.png" width="880" alt="색상화 결과 비교 — 동물" /><br/>
  <img src="assets/result-bird.png" width="880" alt="색상화 결과 비교 — 조류" /><br/>
  <img src="assets/result-place.png" width="880" alt="색상화 결과 비교 — 장소" /><br/>
  <img src="assets/result-human.png" width="880" alt="색상화 결과 비교 — 인물" />
</p>

Baseline은 전반적으로 안정적인 색상 복원을 수행합니다. 경량화 모델은 구조를 단순화했음에도 대상 물체의 경계선이 명확하고 자연스러운 색 복원 결과를 보이며, Baseline 결과에서 관찰되는 고주파 노이즈가 눈에 띄게 줄어듭니다.

---

## 6. 분석

**Residual block 축소는 손해가 없고, 오히려 이득입니다.**
9 → 6 → 4 순으로 연산량이 줄고 처리 속도가 빨라졌으며, PSNR / SSIM / LPIPS도 함께 향상되었습니다. 256×256 색상화 과제에서 9 block 구조는 과도하게 설계되어 있으며, 남는 깊이는 주로 아티팩트를 더할 뿐입니다.

**채널 폭 축소는 속도를 주는 대신 품질을 가져갑니다.**
×0.75는 baseline에 근접하지만 ×0.5, 특히 ×0.25(PSNR 17.885)에서는 성능 저하가 뚜렷합니다. 색 표현력을 담고 있는 것은 깊이가 아니라 채널 폭입니다.

**둘을 함께 줄이는 것이 제한된 예산에서 가장 좋은 절충입니다.**
`4 block + ×0.75` 는 파라미터 3.08M으로 331 FPS를 내면서도 PSNR·SSIM·LPIPS 모두 baseline을 앞섭니다. `4 block + ×0.25` 는 0.34M(baseline의 1/33)으로 355 FPS까지 올라가며, 품질 저하는 크지 않습니다.

**Depthwise Separable Convolution은 이 과제에서는 효과가 없었습니다.**
파라미터 0.73M에도 불구하고 측정된 모델 중 **가장 느렸고**(158 FPS), LPIPS도 전 변형 중 가장 나빴습니다. Depthwise 연산이 GPU에서 메모리 병목에 걸려 FLOPs 절감이 실제 속도로 이어지지 않은 것으로 보입니다.

---

## 7. 한계 및 향후 연구

- **지표와 지각은 다릅니다.** 여러 변형이 PSNR/SSIM/LPIPS에서는 향상되었지만, 사람의 주관적 인식 측면에서는 부자연스러운 결과가 나타나는 경우가 있었습니다. 이 지표들은 인간의 복합적인 시각적 판단을 완전히 반영하지 못합니다. 향후에는 **지각 품질(perceptual quality)** 을 직접 개선하는 방향으로 모델을 보완할 계획입니다.
- **데이터 확대.** 약 12,000장은 이 정도로 열린 과제에 비해 적은 편입니다. 고품질 이미지 데이터의 수와 조명·피사체 다양성을 늘리는 것이 결과 품질 향상의 가장 직접적인 경로입니다.
- **온디바이스 검증.** 양자화·가지치기·지식 증류 등 추가 압축 기법을 적용하고, 데스크톱 GPU FPS가 아니라 실제 모바일·엣지 하드웨어에서의 지연 시간을 측정하는 것이 다음 단계입니다. 목표 응용 분야는 교육용 자료, 문화유산 복원, 모바일 애플리케이션입니다.

---

## 8. 재현 방법

본 실험은 CycleGAN 저자들의 공식 PyTorch 구현체를 기반으로 합니다.

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt
```

**Baseline**

```bash
python train.py \
  --dataroot ./datasets/gray2color \
  --name cyclegan_baseline \
  --model cycle_gan \
  --netG resnet_9blocks \
  --load_size 256 --crop_size 256 \
  --batch_size 16 \
  --n_epochs 50 --n_epochs_decay 50 \
  --lr 0.0002 --beta1 0.5 \
  --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5
```

**경량화 변형.** `models/networks.py` 의 `ResnetGenerator(..., n_blocks=9)` 와 `ngf`(기본 채널 폭)를 수정합니다.

| 변형 | 수정 내용 |
|---|---|
| 6 block / 4 block | `n_blocks = 6` / `n_blocks = 4` |
| Scale ×0.75 / ×0.5 / ×0.25 | `ngf = 48` / `32` / `16` (기본값 `64` 기준) |
| Depthwise | `ResnetBlock` 내부의 `3×3` Conv를 Depthwise Separable Conv + `1×1` bottleneck 으로 교체 |
| 결합 | `n_blocks` 와 `ngf` 를 동시에 조정 |

> **참고.** 현재 이 저장소는 연구 보고서, 그림, 측정 결과를 담고 있습니다. 생성기 수정은 위 설명대로 upstream `networks.py` 에 대한 국소적인 변경입니다.

---

## 9. 인용 및 저자

### 논문

> 강민수†, 강준혁†, 장승기†, 김준화.
> **「CycleGAN 기반 색상화 성능 유지와 처리 속도 향상」**
> *2025년 한국인공지능융합기술학회(KAICTS) 추계학술발표대회*, 2025. 11.
> † 위 저자는 본 논문에 동등하게 기여함.

```bibtex
@inproceedings{kang2025lightweightcyclegan,
  title     = {Performance Preservation and Processing Speed Improvement of CycleGAN-Based Colorization},
  author    = {Kang, Minsu and Kang, Junhyeok and Jang, Seungki and Kim, Junhwa},
  booktitle = {Proceedings of the KAICTS Autumn Conference,
               Korea Artificial-Intelligence Convergence Technology Society},
  year      = {2025}
}
```

### 저자

| 이름 | 소속 |
|---|---|
| 강민수† — *발표자* | 건양대학교 의료인공지능학과 |
| 강준혁† | 건양대학교 의료인공지능학과 |
| 장승기† | 건양대학교 의료인공지능학과 |
| 김준화 | 건양대학교 인공지능학과 |

### 사사

본 연구는 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학사업 지원을 받아 수행되었음 (**2024-0-00047**).

### 참고문헌

1. E. Lin. *Comparative Analysis of Pix2Pix and CycleGAN for Image-to-Image Translation.* Highlights in Science, Engineering and Technology, 39:915–925, 2023.
2. J.-Y. Zhu, T. Park, P. Isola, A. A. Efros. *Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks.* ICCV 2017, pp. 2242–2251.
3. S. Nyamathulla, N. Veeranjaneyulu. *Analysis of Pix2Pix and CycleGAN for Image-to-Image Translation: A Comparative Study.* ICSPCRE 2024, pp. 1–6.
4. R. Steele. *Peak signal-to-noise ratio formulas for multistage delta modulation with RC-shaped Gaussian input signals.* Bell System Technical Journal, 61(3):347–362, 1982.
5. Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP, 13(4):600–612, 2004.
6. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018, pp. 586–595.

### 같은 팀의 관련 연구

- [Gray-to-color-colorization-pix2pix-based](https://github.com/neodle/Gray-to-color-colorization-pix2pix-based) — 쌍 데이터 기반 Pix2Pix로 진행했던 선행 프로젝트
