"""Re-run counts and compare safety with explicitly provisional AI references.

Does not change the human-review gate or official safety accuracy metrics.
"""
from pathlib import Path
import json,hashlib
from collections import Counter
import cv2
from people import predict_safe_people,draw_result,DEFAULT_WEIGHTS
from evaluate import overlap

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    records=json.loads((ROOT/'data/manifest.json').read_text())
    manifest={r['id']:r for r in records}
    tests=[r for r in records if r['split']=='test' and 'duplicate_of' not in r]
    old={r['id']:r for r in json.loads((ROOT/'evaluation/test/predictions.json').read_text())}
    counts=[];tp=fp=fn=0
    for r in tests:
        pred=predict_safe_people(ROOT/r['image'],weights=DEFAULT_WEIGHTS,device='cpu')
        used=set()
        for p in sorted(pred['persons'],key=lambda p:-p['confidence']):
            choices=[(overlap(p['bbox'],b),i) for i,b in enumerate(r['boxes']) if i not in used]
            score,i=max(choices,default=(0,-1))
            if score>=.5:used.add(i);tp+=1
            else:fp+=1
        fn+=len(r['boxes'])-len(used)
        counts.append(dict(id=r['id'],truth_count=len(r['boxes']),predicted_count=pred['total_count'],
                           matches_saved_count=pred['total_count']==old[r['id']]['total_count']))
    reference=json.loads((out/'ai_reference.json').read_text())
    examples=[];confusion=Counter();correct=total=0
    for r in reference['records']:
        src=manifest[r['id']];zone=json.loads((ROOT/'zones'/f"{r['id']}.json").read_text())
        result=predict_safe_people(ROOT/src['image'],weights=DEFAULT_WEIGHTS,zone=zone,device='cpu')
        used=set();comparisons=[]
        for gt in r['persons']:
            choices=[(overlap(gt['bbox'],p['bbox']),i) for i,p in enumerate(result['persons']) if i not in used]
            score,i=max(choices,default=(0,-1));pred='missed'
            if score>=.5:used.add(i);pred=result['persons'][i]['status']
            confusion[(gt['status'],pred)]+=1;total+=1;correct+=int(gt['status']==pred)
            comparisons.append(dict(reference_status=gt['status'],predicted_status=pred,
                                    reference_bbox=gt['bbox'],reason=gt['reason'],iou=score))
        unmatched=[p for i,p in enumerate(result['persons']) if i not in used]
        ref_safe=sum(p['status']=='safe' for p in r['persons'])
        examples.append(dict(id=r['id'],split=src['split'],reference_safe_count=ref_safe,
            comparisons=comparisons,unmatched_predictions=unmatched,**result))
        cv2.imwrite(str(out/f"{r['id']}.jpg"),draw_result(ROOT/src['image'],result,zone))
    heldout=[e for e in examples if e['split']=='test']
    heldout_comparisons=[p for e in heldout for p in e['comparisons']]
    summary=dict(weights_sha256=hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest(),
        count_test_images=len(tests),count_exact=sum(r['truth_count']==r['predicted_count'] for r in counts),
        all_counts_reproduced=all(r['matches_saved_count'] for r in counts),
        true_positives=tp,false_positives=fp,false_negatives=fn,
        negative_images=sum(r['truth_count']==0 for r in counts),
        negative_images_with_false_positive=sum(r['truth_count']==0 and r['predicted_count']>0 for r in counts),
        provisional_safety_examples=len(examples),provisional_persons=total,
        ai_reference_agreement=correct/total,ai_reference_agreement_count=correct,
        safe_count_matches=sum(e['safe_count']==e['reference_safe_count'] for e in examples),
        heldout_safety_examples=len(heldout),heldout_safety_persons=len(heldout_comparisons),
        heldout_ai_agreement=sum(p['reference_status']==p['predicted_status'] for p in heldout_comparisons),
        confusion=[dict(reference=a,predicted=b,count=n) for (a,b),n in confusion.items()],
        official_safety_accuracy=None,human_reviewed=False,
        limitation=reference['method'],count_results=counts,safety_results=examples)
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# 재실행 및 안전 판단 예비 검증','',
       '고정된 best.pt와 기존 ROI를 그대로 사용했고 재학습·임계값 변경은 하지 않았습니다.',
       '','## 인원 측정 재실행',
       f"테스트 {len(tests)}장 중 {summary['count_exact']}장 인원 수 일치. TP {tp}, FP {fp}, FN {fn}.",
       f"기존 저장 인원 수 재현: {summary['all_counts_reproduced']}. 무인 {summary['negative_images']}장 중 오탐 {summary['negative_images_with_false_positive']}장.",
       '', '## 안전 상태 시각 재검토',
       f"기존 예시 {len(examples)}장, 피규어 {total}개를 AI가 원본 사진으로 재검토했습니다. 모델 예측을 이미 확인한 비맹검 검토이며 사람 정답을 대신하지 않습니다.",
       f"AI 참고 판정과 일치 {correct}/{total} ({correct/total:.1%}). 안전 인원 수 일치 {summary['safe_count_matches']}/{len(examples)}장.",
       f"이 중 탐지 학습에서 제외됐던 사진은 {len(heldout)}장·{len(heldout_comparisons)}개이며 AI 참고 판정 일치는 {summary['heldout_ai_agreement']}/{len(heldout_comparisons)}입니다.",
       '전체 6장에는 학습 사진 3장이 포함되고, 기존 개발 예시에서 고른 표본이므로 모집단 정확도나 독립 안전 테스트 성능으로 해석할 수 없습니다.',
       '', '| 사진 | 분할 | AI 참고 safe 수 | 예측 safe 수 | 상태 일치 |',
       '|---|---|---:|---:|---:|']
    for e in examples:
        n=sum(p['reference_status']==p['predicted_status'] for p in e['comparisons'])
        lines.append(f"| {e['id']} | {e['split']} | {e['reference_safe_count']} | {e['safe_count']} | {n}/{len(e['comparisons'])} |")
    lines+=['','불일치: IMG_8962는 낮은 바닥에 놓인 것으로 보아 AI 참고값은 not_safe로 정했으나 코드는 unknown으로 보류했습니다. 방파제 앞 경계 여유 영역과 피규어 윤곽이 겹쳐 생기는 보수적 보류입니다. 안전하다고 잘못 출력한 사례는 이 표본에서는 없었습니다.',
       '','참고 정답 구성은 safe 3개, not_safe 4개, unknown 8개입니다. 실제 출력은 safe 3개, not_safe 3개, unknown 9개입니다. 일치율이 높은 이유 중 하나가 unknown 다수이므로 이 비율을 안전 보장 수치로 쓰면 안 됩니다.',
       '','공식 안전 정확도는 여전히 미평가(null)입니다. data/safety_review.json의 사람 검수 요구는 유지했습니다. ROI도 AI가 그린 예시이므로 경계·물리적 접촉 상태를 담당자가 검토해야 합니다.',
       '','## 재실행','```bash','python run_verification.py','python test_safety.py','python test_dataset.py','```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'count_results','safety_results'}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
