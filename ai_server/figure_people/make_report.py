"""Build the final report from actual saved results, never placeholder metrics."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parent

def main():
    m=json.loads((ROOT/'evaluation/test/metrics.json').read_text())
    baseline=json.loads((ROOT/'evaluation/baseline/metrics.json').read_text())
    safety=json.loads((ROOT/'evaluation/safety/summary.json').read_text())
    predictions=json.loads((ROOT/'evaluation/test/predictions.json').read_text())
    rows=list(csv.DictReader((ROOT/'runs/figure_detector/results.csv').open()))
    best=max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
    errors=[p for p in predictions if p['count_error']]
    examples='\n'.join(f"| {e['id']} | {e['split']} | {e['total_count']} | {e['safe_count']} | {', '.join(p['status'] for p in e['persons'])} |" for e in safety['examples'])
    report=f'''# 실제 실행 결과 보고서

## 변경 파일

사용자 확인 후 기존 `ai_server/danger.py`, `ai_server/fall.py` 두 파일을 삭제했습니다. 새 코드는 `ai_server/figure_people/`에 독립적으로 배치했습니다. 기존에 이미 삭제되어 있던 다른 서버 코드·모델을 복원하거나 변경하지 않았으며 UI·공통 설정·다른 데이터셋은 수정하지 않았습니다.

핵심 파일은 `people.py`(추론/계수/안전), `configure_zone.py`(사진별 구역 지정), `train.py`, `prepare.py`, `export_dataset.py`, `evaluate.py`, `evaluate_safety.py`, `data/dataset.yaml`입니다. 상세 실행법과 과거 코드 분석은 README에 있습니다.

## 데이터와 학습

306장 전체를 분석하고 중복 4장을 제외한 302장을 사용했습니다. 학습 215장, 검증 28장, 테스트 59장입니다. 같은 배치의 연속 촬영은 장면 그룹으로 묶었습니다. 원본 HEIC는 보존했습니다. 19장 무인 사진은 동일 장면이어서 모두 테스트에 함께 두었습니다.

YOLO11n 전이학습, Apple MPS, 최대 60 epochs 중 **실제 {len(rows)} epochs** 완료, batch 8, imgsz 640, seed 42, patience 20입니다. 자동 선택된 optimizer는 AdamW(lr 0.002, momentum 0.9)였습니다. 증강은 학습에만 약한 회전/크기/밝기/수평 반전을 사용했습니다. best.pt는 검증 성능으로 선택했으며, 최고 검증 mAP50-95 기록은 epoch {best['epoch']}의 {float(best['metrics/mAP50-95(B)']):.5f}입니다. 테스트 결과로 모델을 다시 고르지 않았습니다.

## 테스트 59장 실제 결과

| 지표 | 값 |
|---|---:|
| Precision (conf 0.25, match IoU 0.5) | {m['precision']:.6f} |
| Recall (동일 조건) | {m['recall']:.6f} |
| mAP50 | {m['mAP50']:.6f} |
| mAP50-95 | {m['mAP50_95']:.6f} |
| 인원 수 완전 일치율 | {m['count_exact_accuracy']:.2%} |
| 인원 수 MAE | {m['count_mae']:.6f}명 |
| 맞게 탐지한 피규어 | {m['true_positives']} |
| 오탐 / 미탐 | {m['false_positives']} / {m['false_negatives']} |
| 무인 사진 오탐 발생 | {m['negative_images_with_false_positive']} / {m['negative_images']}장 |

사전학습 모델의 동일 테스트 Recall은 {baseline['recall']:.2%}, 인원 수 일치율은 {baseline['count_exact_accuracy']:.2%}, MAE는 {baseline['count_mae']:.4f}명이었습니다. 학습 후와 구분하여 `evaluation/baseline/metrics.json`에 보존했습니다.

계수 오차 이미지: {', '.join(e['id'] for e in errors) if errors else '없음(이번 59장 기준)'}. 정확한 박스 매칭, 신뢰도, 이미지별 오차는 `evaluation/test/predictions.json`에서 확인할 수 있습니다. 오탐/미탐 0은 이 제한된 테스트에 대한 결과이며 다른 환경에서의 오류가 없다는 뜻은 아닙니다.

## 안전 판단

안전 분류 정확도, 안전 인원 수 정확도: **평가 불가(null)**. 독립적인 사람이 검토한 안전 상태 정답이 없습니다. 수치를 임의로 만들지 않았습니다. 아래는 사진별 AI 작성 ROI를 사용한 동작 확인이며, 일부는 학습 장면입니다.

| 사진 | 분할 | 전체 | safe | 피규어별 상태 |
|---|---|---:|---:|---|
{examples}

경계에 앉은 피규어를 무조건 safe로 세지 않고 unknown으로 남깁니다. 테트라포드 쪽 피규어도 전체 수에 포함하되 ROI 밖이면 not_safe입니다. `not_safe`는 위험·추락 확정이 아닙니다. 안전 미탐과 구역 밖 피규어의 잘못된 safe 수 역시 사람 검토 정답이 없어 공식 수치가 없습니다. 이번 예시에서 unknown은 총 {safety['unknown_predictions']}개입니다. 검토 입력 양식은 `data/safety_review.json`입니다.

## 해석과 과적합 한계

학습 첫 epoch의 train box loss는 {float(rows[0]['train/box_loss']):.5f}, 마지막은 {float(rows[-1]['train/box_loss']):.5f}; val box loss는 {float(rows[0]['val/box_loss']):.5f}에서 {float(rows[-1]['val/box_loss']):.5f}로 변했습니다. 학습 곡선은 `runs/figure_detector/results.png`와 CSV에 저장했습니다. best.pt 선택으로 마지막 epoch 성능에만 의존하지 않았습니다.

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
'''
    (ROOT/'REPORT.md').write_text(report)

if __name__=='__main__':main()
