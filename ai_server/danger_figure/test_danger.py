"""python test_danger.py — 샘플 사진 3장으로 위험/안전/빈 사진 판정 확인."""
from pathlib import Path
from danger import predict_danger

S = Path(__file__).parent / "samples"
assert predict_danger(S / "danger.jpg")["danger_count"] >= 1, "위험 사진에서 위험을 못 찾음"
assert predict_danger(S / "sitting_safe.jpg")["danger_count"] == 0, "앉은 안전 인형을 위험으로 오탐"
assert predict_danger(S / "empty.jpg")["danger_count"] == 0, "빈 사진에서 위험 오탐"
print("ok")
