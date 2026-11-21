"""Detection/count evaluation against reviewed labels; never infer truth from folder names."""
from pathlib import Path
import argparse,json
import numpy as np
from ultralytics import YOLO
from people import read_image,draw_result

ROOT=Path(__file__).resolve().parent

def overlap(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    inter=np.prod(np.maximum(hi-lo,0))
    return float(inter/(np.prod(np.array(a[2:])-a[:2])+np.prod(np.array(b[2:])-b[:2])-inter+1e-9))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True);ap.add_argument('--baseline',action='store_true')
    ap.add_argument('--split',default='test',choices=['val','test']);ap.add_argument('--imgsz',default=640,type=int)
    ap.add_argument('--device',default='cpu');args=ap.parse_args()
    model=YOLO(args.weights);out=ROOT/'evaluation'/('baseline' if args.baseline else args.split)
    out.mkdir(parents=True,exist_ok=True)
    records=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']==args.split and 'duplicate_of' not in r]
    results=[];tp=fp=fn=0
    for r in records:
        image=read_image(ROOT/r['image'])
        pred=model.predict(image,classes=[0],conf=.25,iou=.45,imgsz=args.imgsz,device=args.device,verbose=False)[0]
        boxes=pred.boxes.xyxy.cpu().tolist();scores=pred.boxes.conf.cpu().tolist();used=set();matches=[]
        for i in sorted(range(len(boxes)),key=lambda i:-scores[i]):
            candidates=[(overlap(boxes[i],g),j) for j,g in enumerate(r['boxes']) if j not in used]
            v,j=max(candidates,default=(0,-1))
            if v>=.5:used.add(j);matches.append((i,j));tp+=1
            else:fp+=1
        fn+=len(r['boxes'])-len(used)
        result=dict(total_count=len(boxes),safe_count=0,persons=[dict(bbox=b,confidence=s,status='unknown') for b,s in zip(boxes,scores)])
        results.append(dict(id=r['id'],category=r['category'],scene_group=r['scene_group'],
             ground_truth_count=len(r['boxes']),count_error=len(boxes)-len(r['boxes']),matches=matches,**result))
        import cv2
        cv2.imwrite(str(out/f"{r['id']}.jpg"),draw_result(image,result))
    errors=[r['count_error'] for r in results];negative=[r for r in results if r['ground_truth_count']==0]
    summary=dict(n_images=len(results),confidence=.25,match_iou=.5,precision=tp/(tp+fp) if tp+fp else 0,
        recall=tp/(tp+fn) if tp+fn else 0,true_positives=tp,false_positives=fp,false_negatives=fn,
        count_mae=float(np.mean(np.abs(errors))),count_exact_accuracy=float(np.mean(np.array(errors)==0)),
        negative_images=len(negative),negative_images_with_false_positive=sum(r['total_count']>0 for r in negative),
        annotation_provenance='Color proposals visually reviewed by AI; independent human review recommended')
    if not args.baseline:
        metrics=model.val(data=str(ROOT/'data/dataset.yaml'),split=args.split,imgsz=args.imgsz,
            device=args.device,workers=0,project=str(out),name='yolo_metrics',plots=True)
        summary['mAP50']=float(metrics.box.map50);summary['mAP50_95']=float(metrics.box.map)
        summary['ultralytics_precision']=float(metrics.box.mp);summary['ultralytics_recall']=float(metrics.box.mr)
    (out/'predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary)

if __name__=='__main__':main()
