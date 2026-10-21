# 반도체 소자 이상탐지 (PatchCore + YOLOE)

> DACON 반도체 소자 이상탐지 대회 · **Public Score 1위**
> 정상 이미지만으로 학습하는 비지도 이상탐지 — PatchCore에 제로샷 객체 검출(YOLOE)을 결합해 배경 노이즈 문제를 해결

---

## 문제 정의

- **목표**: 반도체 소자 이미지를 정상/이상(0/1)으로 분류
- **데이터**: 흰 배경 보드에 검은 소자가 꽂힌 512×512 이미지. 정상 이미지만 학습에 제공
- **평가**: Binary F1 Score
- **난점**
  1. 소자 **위치·크기가 이미지마다 제각각**
  2. 배경(보드·연결부)에 **이물질·흠**이 자주 존재
  3. 테스트에 **소자가 아예 없는** 이미지가 섞여 있음

---

## 접근: PatchCore

정상 패치의 분포를 메모리뱅크로 저장하고, 테스트 패치가 그 분포에서 **얼마나 먼지(거리)**로 이상을 판정하는 비지도 기법.

```
정상 이미지 → 사전학습 CNN(WideResNet50) 패치 특징 → coreset으로 대표 패치만 저장(메모리뱅크)
테스트 이미지 → 패치별 최근접 거리 → 최댓값 = 이미지 이상점수 → threshold로 0/1
```

기본 PatchCore는 **"이미지 어디든 정상과 다르면 이상"**으로 판단한다. 이 특성이 본 대회의 두 난점(배경 이물질, 소자 부재)과 충돌했고, 아래의 시행착오로 이어졌다.

---

## 실험 과정과 실패 분석

핵심 난점은 두 축이었다 — **(A) 소자 부재를 잡아내기**, **(B) 배경 이물질을 무시하기**. 각 방식이 한 축을 풀면 다른 축이 무너졌다.

| 방식 | (A) 소자 부재 | (B) 배경 이물질 오탐 |
|------|:---:|:---:|
| 전체 이미지 · max 점수 | ❌ 놓침 | ❌ 오탐 |
| 전체 이미지 · max+mean 혼합 | ❌ 놓침 | ❌ 오탐 |
| 전체 이미지 · 엣지 게이트 | ✅ 잡음 | ❌ 오탐 |
| **YOLOE crop + PatchCore** | ✅ 잡음 | ✅ 해결 |

### 1. 전체 이미지 + max 점수
`max`는 "가장 이상한 패치 1개"를 본다. 소자 부재는 **튀는 패치가 없는** 상황이라 점수가 낮게 나와 정상으로 오판했다.

### 2. max + mean 혼합
소자 부재를 잡으려 전체 평균을 섞었으나, 결함 신호가 희석돼 효과가 약했고 부재도 여전히 샜다.

### 3. 엣지 게이트 추가
Canny 엣지 비율이 낮으면 "소자 부재"로 판정 → **부재는 해결**. 그러나 점수는 여전히 **전체 이미지** 기준이라, 정상 소자인데 **배경에 흠**이 있으면 그 패치가 튀어 이상으로 오탐했다.

> 근본 원인: **배경이 점수 계산에 섞이는 것.** 부재를 추정으로 때워도, 배경 오탐은 남았다.

---

## 최종 해법: YOLOE로 소자 영역만 PatchCore

배경을 점수에서 빼는 가장 확실한 방법은 **소자 영역만 보는 것**. 라벨링 없이 소자를 찾기 위해 제로샷 검출 모델 **YOLOE(visual prompt)** 를 사용했다.

```
소자 예시 박스 1개 지정(visual prompt)
        │
        ▼
각 이미지에서 소자 검출
   ├─ 검출 실패 → 소자 부재 → label 1      ← (A) 해결
   └─ 검출 성공 → 소자 박스 crop
                    │
                    ▼
            crop 영역만 PatchCore 점수        ← (B) 배경이 crop 밖 → 무시
                    │
                    ▼
            score > threshold → 0/1
```

**왜 두 문제가 동시에 풀렸나**
- **(A) 소자 부재**: YOLOE가 소자를 못 찾음 = 검출 실패 → label 1. 엣지 추정보다 정확하다 (밝기가 아니라 "소자라는 객체"를 직접 판단).
- **(B) 배경 이물질**: 소자 박스 밖은 애초에 점수에 들어가지 않으므로, 배경이 아무리 지저분해도 무시된다.

### YOLOE를 택한 이유
- 반도체 소자는 일반 객체가 아니라 **텍스트 프롬프트로는 검출 실패** → 소자 예시 박스를 주는 **visual prompt** 방식 채택
- 소자 위치·크기가 제각각이어도 "이렇게 생긴 것"을 **위치 무관**으로 검출
- 라벨링 없이 즉시 적용 가능 (대회 환경에 적합)

### crop 전처리
- 박스에 여유 패딩(10%)을 줘 소자 **가장자리 결함**이 잘리지 않게 함
- 짧은 쪽을 **흰색 여백으로 정사각 패딩** 후 512로 리사이즈 → 비율 왜곡 방지
- 학습·추론에 **완전히 동일한 crop 파이프라인** 적용

---

## 파이프라인

```
[학습]
train 정상 이미지 ──YOLOE crop──> 소자 crop ──WideResNet50 패치특징──> coreset ──> memory_bank.npz

[추론]
test 이미지 ──YOLOE 검출──┬─ 실패 → label 1 (소자 부재)
                          └─ crop → PatchCore 거리 → threshold → 0/1 ──> submission.csv
```

| 구성 | 선택 |
|------|------|
| 특징 추출 | WideResNet50 (layer2 + layer3) |
| 입력 크기 | 512×512 (원본 크기와 동일) |
| 메모리뱅크 압축 | coreset (GPU greedy FPS) |
| 거리 계산 | `torch.cdist` (GPU) |
| 이상점수 | 소자 영역 패치 거리의 max |
| 소자 검출 | YOLOE-11s-seg, visual prompt |

---

## 실행

```bash
pip install -U ultralytics torch torchvision opencv-python numpy

# 1) 소자 crop (학습/추론 각각, 같은 예시 박스 사용)
python yoloe_crop.py --ref train/Train_000.png --img_dir train --out train_crop
python yoloe_crop.py --ref train/Train_000.png --img_dir test  --out test_crop

# 2) 학습: 소자 crop으로 메모리뱅크 구축
python train_patchcore.py --train_dir train_crop --save memory_bank.npz --input_size 512

# 3) 추론: 소자 crop으로 점수 → 제출 CSV + 히트맵
python infer_submission.py --bank memory_bank.npz --img_dir test_crop \
    --absent test_crop/absent_list.txt --threshold 3.0
```

---

## 결과

- **DACON Public Score 1위**
- 소자 부재·배경 이물질 오탐을 모두 제거해, 순수 PatchCore 대비 F1 크게 향상

---

## 배운 점

- **증상이 아니라 원인을 고친다**: 부재·배경 오탐을 각각 땜질(mean 혼합, 엣지 게이트)하기보다, 공통 원인인 "배경이 점수에 섞임"을 제거하자 두 문제가 한 번에 풀렸다.
- **제로샷 검출의 실용성**: 라벨링 없이 YOLOE visual prompt로 ROI를 확보해, 비지도 이상탐지에 "관심 영역만 보기"를 결합할 수 있었다.
- **트레이드오프 기록의 가치**: 각 방식의 실패 패턴을 F1로 검증하며 좁혀간 과정이, 최종 선택의 근거가 되었다.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `yoloe_crop.py` | YOLOE visual prompt로 소자 검출 → 흰색 패딩 crop 저장 |
| `train_patchcore.py` | 소자 crop으로 메모리뱅크(coreset) 구축 |
| `infer_submission.py` | 소자 crop 추론 → 제출 CSV + 히트맵 (부재는 label 1) |


# anomaly_detection_semiconductor
new_train_patchcore.py

new_infer_submission.py

device_region.py

infer_visualize_semicon.py 성능개선 과정

infer_visualize_semicon2.py 성능개선 과정

infer_visualize_semicon3.py 성능개선 과정

