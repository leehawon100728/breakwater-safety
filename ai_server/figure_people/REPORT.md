# 실제 실행 결과 보고서

## 변경 파일

사용자 확인 후 기존 `ai_server/danger.py`, `ai_server/fall.py` 두 파일을 삭제했습니다. 새 코드는 `ai_server/figure_people/`에 독립적으로 배치했습니다. 기존에 이미 삭제되어 있던 다른 서버 코드·모델을 복원하거나 변경하지 않았으며 UI·공통 설정·다른 데이터셋은 수정하지 않았습니다.

핵심 파일은 `people.py`(추론/계수/안전), `configure_zone.py`(사진별 구역 지정), `train.py`, `prepare.py`, `export_dataset.py`, `evaluate.py`, `evaluate_safety.py`, `data/dataset.yaml`입니다. 상세 실행법과 과거 코드 분석은 README에 있습니다.

## 데이터와 학습

306장 전체를 분석하고 중복 4장을 제외한 302장을 사용했습니다. 학습 215장, 검증 28장, 테스트 59장입니다. 같은 배치의 연속 촬영은 장면 그룹으로 묶었습니다. 원본 HEIC는 보존했습니다. 19장 무인 사진은 동일 장면이어서 모두 테스트에 함께 두었습니다.

YOLO11n 전이학습, Apple MPS, 최대 60 epochs 중 **실제 60 epochs** 완료, batch 8, imgsz 640, seed 42, patience 20입니다. 자동 선택된 optimizer는 AdamW(lr 0.002, momentum 0.9)였습니다. 증강은 학습에만 약한 회전/크기/밝기/수평 반전을 사용했습니다. best.pt는 검증 성능으로 선택했으며, 최고 검증 mAP50-95 기록은 epoch 56의 0.99076입니다. 테스트 결과로 모델을 다시 고르지 않았습니다.

## 테스트 59장 실제 결과

| 지표 | 값 |
|---|---:|
| Precision (conf 0.25, match IoU 0.5) | 1.000000 |
| Recall (동일 조건) | 0.967391 |
| mAP50 | 0.992355 |
| mAP50-95 | 0.926452 |
| 인원 수 완전 일치율 | 94.92% |
| 인원 수 MAE | 0.050847명 |
| 맞게 탐지한 피규어 | 89 |
| 오탐 / 미탐 | 0 / 3 |
| 무인 사진 오탐 발생 | 0 / 19장 |

사전학습 모델의 동일 테스트 Recall은 5.43%, 인원 수 일치율은 40.68%, MAE는 1.4746명이었습니다. 학습 후와 구분하여 `evaluation/baseline/metrics.json`에 보존했습니다.

계수 오차 이미지: 231_IMG_9004, 233_IMG_9006, 234_IMG_9007. 정확한 박스 매칭, 신뢰도, 이미지별 오차는 `evaluation/test/predictions.json`에서 확인할 수 있습니다. 오탐/미탐 0은 이 제한된 테스트에 대한 결과이며 다른 환경에서의 오류가 없다는 뜻은 아닙니다.

## 안전 판단

안전 분류 정확도, 안전 인원 수 정확도: **평가 불가(null)**. 독립적인 사람이 검토한 안전 상태 정답이 없습니다. 수치를 임의로 만들지 않았습니다. 아래는 사진별 AI 작성 ROI를 사용한 동작 확인이며, 일부는 학습 장면입니다.

| 사진 | 분할 | 전체 | safe | 피규어별 상태 |
|---|---|---:|---:|---|
| 129_IMG_8986 | train | 2 | 1 | unknown, safe |
| 199_IMG_8780 | test | 3 | 0 | unknown, unknown, unknown |
| 230_IMG_9003 | test | 3 | 2 | unknown, safe, safe |
| 242_IMG_8818 | train | 3 | 0 | unknown, not_safe, not_safe |
| 279_IMG_8891 | train | 3 | 0 | unknown, unknown, not_safe |
| 060_IMG_8962 | test | 1 | 0 | unknown |

경계에 앉은 피규어를 무조건 safe로 세지 않고 unknown으로 남깁니다. 테트라포드 쪽 피규어도 전체 수에 포함하되 ROI 밖이면 not_safe입니다. `not_safe`는 위험·추락 확정이 아닙니다. 안전 미탐과 구역 밖 피규어의 잘못된 safe 수 역시 사람 검토 정답이 없어 공식 수치가 없습니다. 이번 예시에서 unknown은 총 9개입니다. 검토 입력 양식은 `data/safety_review.json`입니다.

## 해석과 과적합 한계

학습 첫 epoch의 train box loss는 1.03945, 마지막은 0.18306; val box loss는 1.27678에서 0.20912로 변했습니다. 학습 곡선은 `runs/figure_detector/results.png`와 CSV에 저장했습니다. best.pt 선택으로 마지막 epoch 성능에만 의존하지 않았습니다.

작은 동일 실내 모형 데이터이고 같은 피규어·배경이 반복됩니다. 장면 그룹 분리에도 환경 일반화와 색상 편향을 배제할 수 없으며, 독립적인 사람의 정밀 박스 검토도 아직 없습니다. 완전히 선 자세, 심한 가림, 4개 이상 밀집, 주황색 비인체 물체, 다른 조명·시점·거리, 새로운 무인 장면을 추가 촬영해야 합니다. 2D 안전 윤곽 판정은 물리적 지지/깊이 판단을 대신하지 않습니다.

## 저장 위치와 실행

- 모델: `runs/figure_detector/weights/best.pt`
- 탐지 결과: `evaluation/test/*.jpg`, `metrics.json`, `predictions.json`
- 안전 예시: `evaluation/safety/*.jpg`, `summary.json`
- 학습 설정/곡선: `runs/figure_detector/args.yaml`, `results.csv`, `results.png`
- 인벤토리/라벨 검토: `data/inventory.csv`, `data/audit.json`, `data/review/`

```bash
cd /Users/ryuahyeon/Desktop/leehawon_latest/ai_server/figure_people
python people.py data/images/230_IMG_9003.jpg --zone zones/230_IMG_9003.json
```

먼저 README의 가상환경 설치 단계를 실행합니다. `predict_safe_people(image, zone=...)`가 독립 JSON 구조를 반환하므로 다른 AI가 이 결과를 받아 합칠 수 있습니다. 실제 웹/카메라/팀원 모델 통합은 하지 않았습니다.
