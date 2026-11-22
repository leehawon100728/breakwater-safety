# 위험 피규어 판정

방파제 모형에서 **위험한 피규어**(테트라포드 위에 올라감 · 틈에 낌 · 바닥에 추락)를 찾습니다. 안전 판정과 전체 인원 수는 `figure_people` 담당이라 여기서는 하지 않습니다. 위험 안에서 끼임/추락을 구분하지도 않습니다(단일 `danger`).

```python
from ai_server.danger_figure import predict_danger
predict_danger("photo.jpg")
# {"danger_count": 2, "persons": [{"bbox": [x1, y1, x2, y2], "confidence": 0.93}]}
```

```bash
pip install -r requirements.txt
python danger.py samples/danger.jpg   # 결과 JSON 출력
python test_danger.py                 # ok 가 나오면 정상
```

- 입력은 JPEG/PNG 경로 또는 BGR NumPy 배열이고 `bbox` 는 그 입력의 픽셀 좌표입니다. HEIC 는 JPEG 로 바꿔서 넣습니다.
- 카메라 위치를 고정하거나 위험 구역을 그릴 필요가 없습니다. 모델이 모양과 주변을 보고 직접 판단합니다.

## 모델

YOLO11n(`best.pt`, 5.5MB)을 2클래스로 학습했습니다: `danger`(위험), `safe`(방파제 윗면에 앉아 있는 안전). `safe` 는 앉은 인형을 위험으로 오인하지 않게 하는 학습용이라 결과에는 넣지 않습니다.
학습 데이터는 위험 154장, 안전 27장, 인형 없는 사진 8장과 위험 사진에서 인형을 지워 만든 합성 빈 사진 약 20장이고, 라벨은 주황색 자동 검출 뒤 일부만 눈으로 확인했습니다.

## 한계

- 검증 사진이 학습 사진과 같은 장소·날에 찍은 것이라 검증 점수(위험 30장 82개 검출, 안전 6장·빈 사진 8장에서 위험 0)는 실제보다 높을 수 있습니다.
- 안전 사진이 한 각도에서 찍은 앉은 인형뿐이라, 다른 각도(반대편 카메라 등)와 서 있는 안전 자세는 검증하지 못했습니다.
- 새 촬영 환경에서 오탐·미탐이 있으면 그 사진을 위험/안전/빈 사진으로 나눠 추가해 다시 학습해야 합니다.
