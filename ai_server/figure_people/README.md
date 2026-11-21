# 방파제 모형 피규어 인원 측정·안전 구역 판단

주황색 사람 모양 피규어를 단일 클래스 `person_figure`로 학습하고, 사진 한 장에서 전체 인원과 안전 인원을 반환합니다. 위험 구역의 피규어도 전체 인원에 포함합니다. 위험·추락 판정, 카메라, 웹 화면 통합은 구현하지 않습니다.

## 변경 범위

사용자가 후속 메시지에서 삭제를 명시적으로 확인한 기존 `ai_server/danger.py`, `ai_server/fall.py`를 삭제했습니다. 현재 프로젝트 폴더에는 인원 측정·안전 판단용 Python 코드와 가중치가 없었습니다. Git HEAD에서 확인한 예전 `people_counter.py`는 Teachable Machine의 0~4명 장면 분류였고, `person_detector.py`는 일반 COCO person + ByteTrack 추적 방식이었습니다. 이 파일들과 여러 기존 서버 파일은 이번 작업 시작 전에 이미 삭제된 상태였으며 복원·변경하지 않았습니다. 새 코드는 폴더/장면 분류가 아니라 학습된 피규어 박스 수를 계산하고, 이미지 입력에 필요 없는 영상 추적 상태를 두지 않습니다. `index.html`의 인원 표시는 시연용 JavaScript였으며 수정하지 않았습니다. 새 모듈은 프로젝트의 `ai_server/figure_people/`에 독립적으로 배치합니다. 공통 설정과 웹 시작 스크립트는 변경하지 않습니다.

## 실제 데이터 점검

| 폴더 | 원본 수 | 사진에서 검토한 피규어 수 |
|---|---:|---|
| 사람1 61 | 61 | 각 1개 |
| 사람2 81 | 81 | 각 2개 |
| 사람3 77 | 77 | 각 3개 |
| 위험+안전 68 | 68 | 64장은 3개, 4장은 2개 |
| 사람 X 19 | 19 | 각 0개 |

모두 HEIC입니다. EXIF 회전을 반영한 원본 해상도는 3024×4032 239장, 4284×5712 67장입니다. 원본을 변경하지 않고 긴 변 1280 JPEG 사본을 생성했습니다. 기존 YOLO 라벨은 없었습니다. 색상으로 박스를 제안한 뒤 **AI가 306장 전체의 검토용 연락판 16개를 시각 검토**했습니다. 독립적인 사람의 정밀 라벨 검수는 아직 받지 않았으므로 성능은 이 라벨을 기준으로 한 개발 평가입니다. 변환 이미지에서 박스 크기는 너비 59–185px, 높이 76–184px입니다. 앉거나 눕고 기댄 자세, 화면 끝에 잘린 피규어가 포함됩니다.

`IMG_8814`–`IMG_8817`은 사람2와 위험+안전에 중복되어 있습니다. SHA-256으로 확인한 동일 파일 4개를 학습·평가에서 제외했습니다. 원본 306장, 고유 302장입니다. 같은 배치의 연속 사진과 나중에 재촬영한 유사 배치를 함께 묶어 **train 215 / val 28 / test 59**, 19개 장면 그룹으로 분할했습니다. 무작위 프레임 분할은 하지 않았습니다. 실내 모형·배경은 전체 데이터에 공통이므로 새 촬영 환경 일반화 검증을 대신하지 않습니다.

빈 장면 19장은 모두 매우 유사한 한 촬영 묶음이므로 테스트에 함께 보관했습니다. 따라서 이번 train에는 별도의 무인 사진이 없습니다. 19장의 네거티브 테스트를 유지하면서 네거티브 학습도 추가하려면 다른 시점·조명에서 새 무인 사진이 필요합니다. 원본 5개 폴더는 분석과 학습/검증/평가 전체에 모두 활용했습니다.

`data/inventory.csv`에 이미지별 수·장면 그룹·분할·중복·원본 경로가 있고, `data/audit.json`에 해상도·중복 등 점검 결과가 있습니다. `data/review/`는 정답 박스 검토용이며 모델 예측 결과가 아닙니다.

## 설치 및 실행

Python 3.11 이상과 가상환경을 사용합니다. 실제 작업 환경은 Python 3.11, Ultralytics 8.4.216입니다.

```bash
cd /Users/ryuahyeon/Desktop/leehawon_latest/ai_server/figure_people
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python people.py data/images/230_IMG_9003.jpg --zone zones/230_IMG_9003.json
```

출력은 `predictions/230_IMG_9003.json`과 `.jpg`입니다. HEIC·JPEG·PNG 경로 또는 BGR uint8 NumPy 배열을 입력할 수 있습니다. ROI 없이 실행하면 인원 수는 정상 계산하고 모든 피규어의 안전 상태는 `unknown`으로 남깁니다. 학습된 `best.pt`가 없으면 명시적인 오류를 내며 다른 사람용 모델로 조용히 대체하지 않습니다.

```python
from ai_server.figure_people import predict_safe_people, count_people, classify_safety

result = predict_safe_people(
    image,
    zone="ai_server/figure_people/zones/230_IMG_9003.json",
)
# {"total_count": ..., "safe_count": ...,
#  "persons": [{"bbox": [x1,y1,x2,y2], "confidence": ..., "status": ...}]}
total = count_people(image)
# 이미 탐지한 단일 피규어의 안전 상태만 별도로 구할 수도 있습니다.
# status = classify_safety(image, bbox, zone="zones/my_photo.json")
```

위 import는 `leehawon_latest`를 Python 작업 디렉터리/경로로 설정했을 때 사용합니다. 다른 팀원 AI와의 실제 통합은 수행하지 않았습니다. 나중에 원본 이미지의 같은 좌표계에서 `persons` 결과를 합치면 됩니다. `not_safe`는 안전 구역 밖이라는 의미이며 위험·추락 사건의 확정이 아닙니다. 입력 경로는 EXIF 회전을 적용하고, NumPy 배열은 이미 회전된 BGR이라고 가정합니다. bbox는 그 입력 이미지의 픽셀 좌표입니다.

## 안전 구역 설정과 기준

방법 비교: 이미지별 ROI는 작은 데이터에서 구조 경계를 명시하고 시점별 재설정이 쉬워 이번 기본 방식으로 선택했습니다. 방파제 segmentation은 추가 경계 정답과 별도 학습이 필요합니다. 흰색 임계값만의 OpenCV 분리는 같은 흰색인 바닥·벽·방파제 측면을 혼동하므로 사용하지 않습니다. **YOLO + 사람이 지정할 수 있는 구조 ROI + 피규어 윤곽 보조**를 사용합니다.

```bash
python configure_zone.py /absolute/path/to/photo.HEIC --output zones/my_photo.json
python people.py /absolute/path/to/photo.HEIC --zone zones/my_photo.json
```

열린 창에서 흰색 방파제의 **평평한 상면 경계만** 순서대로 클릭합니다. 오른쪽 클릭은 취소, Enter는 저장, Esc는 종료입니다. 흰색 바닥·측벽·테트라포드는 포함하지 않습니다. ROI는 정규화된 다각형, 이미지 크기, 회전 적용 픽셀의 SHA-256으로 사진에 결합됩니다. 다른 사진에 잘못 재사용하면 오류가 발생합니다. 같은 사진도 크기를 바꾸거나 다시 압축하면 ROI를 다시 설정해야 합니다. 향후 고정 카메라 영상은 시점 고정·보정 정책을 별도로 설계해야 하며 현재 코드는 이미지 전용입니다.

YOLO가 찾은 박스 안에서 주황색 윤곽을 추출해 방파제와 겹치는 위치를 검사합니다. 색상은 사람 수를 만들거나 안전 구역을 찾는 데 쓰이지 않습니다. 색상 윤곽이 충분하지 않으면 `unknown`입니다. 윤곽을 살짝 확장하여 압축·마스크 오차에 여유를 둡니다.

- `safe`: 전체 윤곽이 경계 여유 영역을 제외한 방파제 내부에 있음.
- `not_safe`: 전체 윤곽이 방파제와 경계 여유 영역 밖에 있음.
- `unknown`: 가장자리에 걸침, 일부만 안쪽/바깥쪽, 색상 윤곽 불명확, 사진에 잘림, ROI 미설정/미검토.

박스 하단 중심점을 사용하지 않습니다. 가장자리에 앉아 다리가 밖으로 나온 경우는 자동으로 안전 처리하지 않습니다. 이 기준은 2D 투영 위치를 보수적으로 분류하며 물리적인 지지·안정성을 측정하지는 않습니다. 높이 때문에 다른 구조물의 피규어가 ROI에 겹쳐 보이는 경우나 강한 가림에서는 별도 접촉/깊이 정답이 필요합니다. 제공된 `zones/` 6개는 AI가 사진을 보고 그린 예시이며 현장 담당자의 경계 검토가 필요합니다.

## 재학습 및 평가

원본을 다시 준비할 때만 `prepare.py`를 실행합니다. 이는 기존 manifest 검토 상태를 초기화합니다. 새 사진/변경 라벨은 정밀 검토 후 `annotation_review.json`의 이미지·박스 지문을 갱신해야 합니다. `export_dataset.py`의 장면 배정은 이번 306장 전용이므로 새 데이터에는 직접 장면 그룹을 다시 지정해야 합니다.

```bash
python export_dataset.py
python train.py --model yolo11n.pt --epochs 60 --imgsz 640 --batch 8 --device auto
python evaluate.py --weights runs/figure_detector/weights/best.pt --split test --device cpu
python evaluate_safety.py
python test_safety.py
```

`data/dataset.yaml`에 단일 클래스가 정의되어 있습니다. 폴더를 옮긴 뒤에는 `python export_dataset.py`를 실행해 데이터 경로를 갱신합니다. 재학습은 기존 runs 폴더를 덮어쓰지 않고 새 실행 폴더를 만듭니다. 새 실행의 가중치는 `--weights` 또는 `FIGURE_WEIGHTS`로 지정합니다.

실제 학습 설정: YOLO11n COCO 사전학습 전이학습, 최대 60 epochs, imgsz 640, batch 8, Apple MPS, seed 42, patience 20, workers 0. 회전 ±10도, 이동 5%, 크기 15%, 수평 반전, 약한 HSV 변화만 학습에 적용하고 mosaic/mixup/상하반전은 끕니다. val과 test에는 학습 증강을 적용하지 않습니다. GPU 일부 연산은 완전한 결정성이 없어 seed가 같아도 수치가 조금 달라질 수 있습니다.

전체 인원은 클래스 0의 신뢰도 ≥0.25 탐지 개수입니다. NMS IoU 0.45로 중복을 줄입니다. 박스·색상 후보 개수나 폴더명으로 인원을 결정하지 않습니다. 기본 추론 크기는 학습·평가와 같은 640입니다.

탐지·계수 지표는 `evaluation/test/metrics.json`, 이미지별 오차와 매칭은 `evaluation/test/predictions.json`, 시각화는 `evaluation/test/*.jpg`에 저장합니다. 신뢰도 0.25에서 Precision/Recall과 계수 정확도를 계산하며, 별도로 Ultralytics의 AP 평가 지표를 저장합니다. 서로 다른 신뢰도 처리의 Precision/Recall은 구분해 기록합니다. 기준 사전학습 모델의 결과는 `evaluation/baseline/`에 있습니다.

안전 예시 출력은 `evaluation/safety/`입니다. **안전 정답이 없으므로 안전 분류 정확도와 안전 인원 수 정확도는 null**입니다. 예시 6장은 train/test가 섞인 동작 확인 자료이며 독립적인 안전 평가가 아닙니다. `data/safety_review.json`의 bbox별 `status`를 사람이 판정하고 `reviewed: true`, `reviewer: "human"`으로 설정하면 `evaluate_safety.py`가 안전 정확도·안전 수 일치율·안전 미탐·구역 밖 피규어의 잘못된 safe 분류를 계산합니다. detection 누락은 안전 분류 오답으로 포함합니다.

## 추가 촬영

완전히 서 있는 피규어, 가림이 심한 피규어, 4개 이상 밀집·겹침, 다양한 회전의 누운 자세, 가장자리 안팎의 애매한 지지 상태, 다른 시점·거리·조명, 주황색 비인체 물체, 새 무인 장면을 추가해야 합니다. 사진별 방파제 경계와 안전 상태도 함께 기록해 독립적인 사람의 검토를 받는 것이 다음 단계입니다. 현재 지표는 동일 실내 모형의 제한된 배치 평가이며 실제 사람·야외 방파제 성능을 뜻하지 않습니다.

Ultralytics API 근거: https://docs.ultralytics.com/modes/train/ , https://docs.ultralytics.com/modes/val/

## 추가 검증

`python run_verification.py`는 CPU로 테스트 59장 인원 수를 다시 계산하고, 구역이 설정된 6장·피규어 15개의 안전 상태를 별도의 AI 시각 재검토 참고값과 비교합니다. 결과는 `verification/REPORT.md`, `verification/results.json` 및 사진별 JPEG에 저장합니다. 6장에는 학습 사진 3장이 포함되며, 예측을 이미 본 뒤의 AI 검토이므로 공식 안전 정확도나 독립 평가로 취급하지 않습니다. 사람 정답을 요구하는 기존 `evaluate_safety.py`의 검수 조건은 그대로 유지합니다.
