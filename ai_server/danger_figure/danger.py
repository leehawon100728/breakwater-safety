"""위험 상태의 피규어를 찾는다 (테트라포드 위·틈에 낀·바닥에 추락한 피규어).

학습 클래스는 danger(위험) / safe(방파제 윗면에 앉아 있는 안전)이고, 안전은 위험 오탐을 줄이려는 학습용이라
결과에는 위험만 담는다. 안전 판정·전체 인원은 figure_people 이 맡는다.

    from ai_server.danger_figure import predict_danger
    predict_danger("photo.jpg")  # {"danger_count": 2, "persons": [{"bbox": [x1,y1,x2,y2], "confidence": 0.93}]}

입력: JPEG/PNG 경로 또는 BGR uint8 NumPy 배열 (bbox 는 그 입력의 픽셀 좌표). HEIC 는 JPEG 로 바꿔서 넣는다.
"""
import json
import sys
from pathlib import Path

import cv2

WEIGHTS = Path(__file__).with_name("best.pt")
_model = None


def _load():
    global _model
    if _model is None:
        if not WEIGHTS.exists():
            raise FileNotFoundError(f"가중치 없음: {WEIGHTS}")  # 다른 모델로 조용히 대체하지 않음
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model


def predict_danger(image, conf=0.5):
    if isinstance(image, (str, Path)):
        path = image
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없음: {path}")
    model = _load()
    persons = []
    for b in model.predict(image, conf=conf, verbose=False)[0].boxes:
        if model.names[int(b.cls)] == "danger":
            persons.append({"bbox": [round(float(v), 1) for v in b.xyxy[0]], "confidence": round(float(b.conf), 3)})
    return {"danger_count": len(persons), "persons": persons}


if __name__ == "__main__":
    print(json.dumps(predict_danger(sys.argv[1]), ensure_ascii=False, indent=2))
