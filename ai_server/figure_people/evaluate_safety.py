"""Create image+JSON safety examples; score only explicitly reviewed safety truth."""
import json
from pathlib import Path
import cv2
from people import predict_safe_people,read_image,draw_result,DEFAULT_WEIGHTS
from evaluate import overlap

ROOT=Path(__file__).resolve().parent

def main():
    manifest={r['id']:r for r in json.loads((ROOT/'data/manifest.json').read_text())}
    reviews=json.loads((ROOT/'data/safety_review.json').read_text())
    out=ROOT/'evaluation/safety';out.mkdir(parents=True,exist_ok=True)
    examples=[];matched_correct=annotated_total=count_correct=images_scored=false_safe=missed_safe=0
    for truth in reviews:
        r=manifest[truth['id']];zone=json.loads((ROOT/'zones'/f"{r['id']}.json").read_text())
        result=predict_safe_people(ROOT/r['image'],weights=DEFAULT_WEIGHTS,zone=zone)
        example=dict(id=r['id'],split=r['split'],safety_truth_reviewed=truth['reviewed'],**result)
        examples.append(example)
        cv2.imwrite(str(out/f"{r['id']}.jpg"),draw_result(ROOT/r['image'],result,zone))
        (out/f"{r['id']}.json").write_text(json.dumps(result,indent=2))
        if not truth['reviewed']:continue
        if truth.get('reviewer')!='human':raise ValueError('Safety truth requires independent human review')
        persons=truth.get('persons',[])
        if not persons or any(p['status'] not in {'safe','not_safe','unknown'} for p in persons):
            raise ValueError('Incomplete safety truth')
        images_scored+=1;count_correct+=int(result['safe_count']==sum(p['status']=='safe' for p in persons))
        used=set()
        for gt in persons:
            annotated_total+=1
            choices=[(overlap(gt['bbox'],p['bbox']),i) for i,p in enumerate(result['persons']) if i not in used]
            score,i=max(choices,default=(0,-1));pred=None
            if score>=.5:used.add(i);pred=result['persons'][i]['status']
            matched_correct+=int(pred==gt['status'])
            false_safe+=int(pred=='safe' and gt['status']=='not_safe')
            missed_safe+=int(gt['status']=='safe' and pred!='safe')
    summary=dict(examples=examples,reviewed_images=images_scored,
        safety_classification_accuracy=matched_correct/annotated_total if annotated_total else None,
        safe_count_exact_accuracy=count_correct/images_scored if images_scored else None,
        outside_figures_incorrectly_safe=false_safe if images_scored else None,
        safe_figures_missed=missed_safe if images_scored else None,
        unknown_predictions=sum(p['status']=='unknown' for e in examples for p in e['persons']),
        note='Illustrative examples include train scenes; not an independent safety benchmark. Null metrics mean no human-reviewed safety truth.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
