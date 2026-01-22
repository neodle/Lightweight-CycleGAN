# Maintaining-CycleGAN-based-colorization-performance-and-improving-processing-speed 
# (CycleGAN 기반 색상화 성능 유지와 처리 속도 향상)
---


# CONTENTS


* **Introduction** : 프로젝트 연구 배경

  
* **Purpose & Method** : 연구 목적 및 방법

  
* **Dataset & Data preprocessing** : 데이터 및 전처리  


* **CycleGAN Baseline** : 베이스라인 모델(CycleGAN)

  
* **Lightweight CycleGAN** : CycleGAN 경량화


* **Experiment Result** : 연구 결과


* **future works** : 향후 계획
---


# Motivation

기존 지도학습 기반의 모델인 Pix2Pix의 학습되지 않은 이미지에 대한 색상화 성능이 좋지 않음

이러한 한계를 이겨내기 위해서 비지도 학습 모델인 생성 적대적 신경망인 GAN이 대두되어 왔으며 그중에도 CycleGAN이 대두됨

하지만 CycleGAN도 모델 자체가 너무 무거우며, 실시간성이 너무 부족하다는 단점이 존재함

이에 본 프로젝트에서는 기존의 비지도 학습 기반의 CycleGAN보다 시간과 비용 측면에서 효율적인 딥러닝을 이용한 색상화 방식에 대해 연구를 진행
  
자연스러운 복원 결과를 위해 시각적 유사성을 평가하는 성능 평가 지표 뿐만 아니라 사용자 주관적 평가도 함께 고려함

이를 통해 역사적 기록물, 사진 복원, 교육용 자료등 다양한 분야에 활용할 수 있으며, 더나아가 개인의 소중한 기억의 한 조각을 찾는 효과를 기대할 수 있음 

<p align="center">
  <img src="<img width=975 img height="485 alt="image" src="https://github.com/user-attachments/assets/6f780525-e354-43ab-acee-b5bcc3c866ce" />"
</p>


--- 

# **Project object**

**프로젝트 목표:** 딥러닝 모델을 활용한 흑백 사진을 자연스럽고 실사에 가까운 컬러 이미지로 복원하는 AI 시스템 구현


--- 


# **Flowchart** 

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d278af9-5f86-477f-bb96-d5dcee76fed9" width="700" />
</p>

---


# Dataset

## 데이터 전처리 과정 및 순서

I. 데이터 셋 수집

- 다양한 쌍 이미지 기반의 이미지 데이터 셋 수집 (다양한 풍경, 인물, 동물, 꽃, 음식 등)

II. 데이터 EDA (탐색적 데이터 분석) 

-  EDA 이후, 최종 데이터 셋 결정 
-  최종 데이터 셋 결정 후 프로젝트에 쓰이는 데이터 셋 생성
-  **데이터셋 구성** [Train (color 1500 : gray-scale 1500)] : [Validation (color 100 : gray-scale 100)] : Test [gray-scale 150]

III. EDA 분석 

- 컬러 이미지 데이터와 흑백 이미지 데이터 높이(Height)와 너비(width) 분포

| Gray-scale img EDA | Color img EDA |
|---------|---------|
| <img src= "https://github.com/user-attachments/assets/c6f8657a-f4f0-4979-ba07-77073cb719b3" width="550" height="420" /> | <img src= "https://github.com/user-attachments/assets/f89caa2f-705a-43a0-98b6-4a7c5fbc8bd9" width="550" height="420" /> |

- Color 이미지 데이터의 RGB channel 분포

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd87f8b7-a9b7-462c-aece-c277cec31b6b" width="700" />
</p>

**컬러 이미지**와 **흑백 이미지** 모두 **150×150 크기**의 정사각형 이미지 데이터

**컬러 이미지**에서 **R, G, B** 세 채널 모두 대체로 균일하게 분포되어 색상의 균형이 잘 맞춰져있음

IV. 데이터 전처리 

- 모든 이미지가 동일한 크기를 가지므로 모델 학습에 앞서 별도의 리사이징 과정 및 이미지 데이터 전처리 과정이 필요하지 않음

---

# Model Architecture  

**Pix2Pix** 

## Pix2Pix - U-Net based Generator + PatchGAN Discriminator

<p align="center">
<img src="https://github.com/user-attachments/assets/92efdf19-051b-421b-9006-aa8322b542e5" width="700" height="400" />
</p>

## Pix2Pix 모델 선정 배경 및 구조

## Pix2Pix 모델 선정 배경

**1. 흑백 이미지 → 컬러 이미지 색상화 픽셀 단위의 정밀한 매핑이 필요한 image-to-image translation 문제** 

**2. U-Net 구조는 인코더(압축)와 디코더(복원) 사이에 skip connection이 있음 → 저수준의 공간 정보를 고해상도로 정확히 복원할 수 있음**

**3. U-Net 구조 덕분에 색상화 시 중요한 윤곽선, 질감, 경계 정보가 잘 유지됨**

**4. 일반적인 Discriminator는 이미지 전체를 보고 Real/Fake를 구분**

**5. 색상화 문제에서는 국소 영역(작은 패치)의 색 배치나 질감이 자연스러운지가 더 중요함**

**6. PatchGAN은 전체 이미지를 하나로 처리하지 않고, 공통된 소형 CNN 필터를 반복 적용하여 패치를 평가함 → 모델 파라미터 수가 적고 속도도 빠르며, 학습이 안정적임**

**<결론>** U-Net based Generator와 PatchGAN Discriminator가 결합된 구조인 pix2pix 모델을 선정 

## U-Net based Generator

**1. 인코더(encoder) 부분에서 입력 이미지가 점점 축소되며 특징을 추출함** 

**2. 디코더(decoder) 부분에서 이미지에 대한 해상도를 다시 복원함**

**3. 점선으로 이어진 화살표는 입력 정보와 출력 정보를 직접 연결하는 U-Net의 핵심인 스킵 커넥션이며, 저수준 정보를 고수준 레이어에 직접 전달해 디테일 유지함**

## Patch GAN Discriminator

**1. U-Net 기반의 Generator (T) 부분에서 생성한 결과 이미지인 T(x)와 정답 이미지인 Ground Truth y 두개의 이미지 이용해 판별에 사용** 

**2. CNN 기반으로 점점 공간의 크기를 줄여가며 특징 추출을 하며, P1, P2, P3, P4의 여러 단계에서 T(x)와 Ground Truth y 쌍의 정합성을 확인함**

**3. 마지막 출력에서 실제 이미지인지, Generator가 만든 가짜 이미지인지 판단**

**4. 판별 결과를 바탕으로 Generator는 더 정답과 비슷한 T(x)를 만들도록 학습됨**

---

# Model Analysis 

## 모델의 문제점

<p align="center">
<img src= "https://github.com/user-attachments/assets/75ccf1ca-92ff-4786-a7ea-92f09f820bd6" width="700" height="400" />
</p>

**1.** 결과 이미지(Result Image)가 원본 이미지와 비교했을 때 품질이 좋지 않음

**2.** 성능지표 또한 수치가 높게 나오지 않음

**성능지표(Evaluation metrix)**

**1. PSNR** - Peak Signal-to-noise ratio

**2. SSIM** - Structural Similarity Index Measure

**3. LPIPS** – Learned Perceptual Image Patch Similarity

**4. FID** - Fréchet inception distance

## 모델의 성능 저하의 원인

**1. 하이퍼 파라미터 조정** : PSNR, SSIM이 모두 낮고 LPIPS, FID이 모두 높다면 학습이 부족하다는 의미임 (현재 epoch 10회)

**2. G_LOSS** - 학습 결과 Discriminator의 loss가 일정하게 낮은 반면, Generator의 loss는 epoch이 증가하여도 감소하지 않고 높은 수준(10 ~ 12)에서 진동함

| Evaluation Matrix | LOSS Graph |
|---------|---------|
| <img src= "https://github.com/user-attachments/assets/358dd685-c8a1-47c1-81fa-5069cb44206d" width="550" height="420" /> | <img src= "https://github.com/user-attachments/assets/f6f4919b-dbac-46fc-a46d-7c7078365128" width="550" height="420" /> |

# Model Improvement

**1. 하이퍼 파라미터 조정** : epoch 50회, early-stopping, Learning rate 조정

**2. G_LOSS** - Generator가 충분히 학습할 시간을 확보, 8번째 epoch 이후부터 Discriminator를 학습시킴 -> 초반에 Discriminator 학습을 막아 Generator가 더 안정적으로 학습 초기화 가능하게 함

| Improve Evaluation Matrix | Improve LOSS Graph |
|---------|---------|
| <img src= "https://github.com/user-attachments/assets/a6e3338f-b729-4d04-b2ea-1c3ab0f3ae26" width="550" height="420" /> | <img src= "https://github.com/user-attachments/assets/31b2f404-0173-4082-9645-940656137a9a" width="550" height="420" /> |

# Image Colorization Performance Comparison

<p align="center">
<img src= "https://github.com/user-attachments/assets/fc5eb83b-0ff9-4a00-bc1e-e18c08d8fde8" width="900" height="400" />
</p>

**생각할 점: Iil posed problem**
모델의 성능이좋다고 결과 이미지의 성능이 좋아지는것이 아님

따라서 사용자의 주관적 평가에 초점을 두기도 함

이번에 시도한 프로젝트에서는 운이 좋게도 성능이 가장 좋은 모델에서 나온 결과 이미지가 사용자 주관적 평가에서도 좋은 평가를 받음

<p align="center">
<img src= "https://github.com/user-attachments/assets/8ed6cdff-9e64-4666-8568-a73cf7184723" width="900" height="200" />
</p>

# Service deploy

<p align="center">
<img src= "https://github.com/user-attachments/assets/9ab66b7d-0d88-4310-9efc-6514ff7f64ef" width="800" height="400" />
</p>

파이썬 기반의 마이크로 웹 프레임워크인 **FLASK** 를 이용해 웹페이지를 개발하여 서비스 방안을 구상

---

# 프로젝트의 한계 및 Future works

---
**프로젝트의 한계**

<p align="center">
<img src= "https://github.com/user-attachments/assets/8e719aa3-b714-458c-ace8-d7eac8a301e7" width="800" height="200" />
</p>

* 학습되지 않은 이미지 데이터에 대한 색상화를 하지 못함 -> 학습 이미지 데이터 부족


**1. 학습 데이터 추가 및 증강 시도**

학습 데이터의 양과 다양성을 확장하여 모델의 일반화 성능 향상 그리고 보다 다양한 장면, 조명, 피사체 등을 포함하는 이미지 수집하여 복잡한 컬러화 상황 대응력 강화

**2. 평가 지표 및 주관적 평가 보완**

사용자 주관적 평가 데이터를 축적하여 실제로 서비스를 이용하는 사용자 관점의 품질 평가 반영

---

# 기대효과

**1. 역사적 가치의 복원과 대중화**

흑백 사진·영상을 컬러로 전환함으로써 과거가 더욱 현실감 있게 다가오며, 역사의 생생한 감각을 느낄 수 있음

단절된 시간 속 인물과 공간에 감정을 이입하게 함으로써 공감과 관심을 이끌어내어 사람들의 역사적 거리감을 해소할 수 있음

**2. 심리적·정서적 기대효과**

개인의 추억, 기억 또는 감정을 자극하여 향수와 감동을 유도할 수 있음

**3. 기록 보존 및 연구 자료 활용**

복원된 컬러 이미지들이 다양한 분야에서 인공지능 연구·개발에 유용한 자료가 됨
