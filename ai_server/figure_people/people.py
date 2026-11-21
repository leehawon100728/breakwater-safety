"""Independent still-image figure counting and conservative safe-zone inference."""
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'runs' / 'figure_detector' / 'weights' / 'best.pt'

def read_image(image):
    """Paths -> EXIF-oriented BGR; ndarray inputs must already be oriented BGR."""
    if isinstance(image, (str, Path)):
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(image)).convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must be an image path or HxWx3 BGR ndarray')
    if image.dtype != np.uint8 or not image.size:
        raise ValueError('image must be nonempty uint8 BGR')
    return image

def image_digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()

def zone_masks(image, zone):
    """ROI must be explicitly approved and bound to these exact oriented pixels."""
    if zone is None or not zone.get('reviewed', False): return None
    h,w = image.shape[:2]
    if zone.get('image_size') != [w,h] or zone.get('image_sha256') != image_digest(image):
        raise ValueError('ROI image/size mismatch; configure this image separately')
    points=np.asarray(zone.get('polygon',[]),dtype=np.float64)
    if points.ndim != 2 or points.shape[0]<3 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('polygon must have at least 3 finite normalized points')
    if np.any(points<0) or np.any(points>1): raise ValueError('polygon coordinates must be in [0,1]')
    contour=np.rint(points*[w-1,h-1]).astype(np.int32)
    if cv2.contourArea(contour)<4 or not cv2.isContourConvex(contour):
        raise ValueError('Use an ordered convex polygon for the flat deck top')
    mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[contour],255)
    margin=float(zone.get('margin_fraction',0.02))
    if not 0<=margin<=0.25: raise ValueError('invalid margin_fraction')
    radius=max(1,round(min(h,w)*margin))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    inner=cv2.erode(mask,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    outer=cv2.dilate(mask,kernel)
    return inner,outer

def judge_safety(bbox, masks, image_shape, frame=None):
    """Use the entire detection footprint, never the bottom-center alone.

    Fully within inset deck -> safe; fully outside expanded deck -> not_safe.
    Boundary/partial overlaps, missing zones, and truncated objects -> unknown.
    This is a conservative 2-D ROI rule, not a physical support classifier.
    """
    if masks is None: return 'unknown'
    h,w=image_shape[:2]
    x1,y1,x2,y2=bbox
    if not np.isfinite(bbox).all() or x2<=x1 or y2<=y1: return 'unknown'
    if x1<=0 or y1<=0 or x2>=w-1 or y2>=h-1: return 'unknown'
    x1,y1=int(np.floor(x1)),int(np.floor(y1))
    x2,y2=int(np.ceil(x2)),int(np.ceil(y2))
    inner,outer=masks
    support=np.ones((y2-y1,x2-x1),bool)
    if frame is not None:
        # Color assists footprint localization ONLY within a YOLO-confirmed box.
        # It never creates detections or defines the deck geometry.
        crop=frame[y1:y2,x1:x2]
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        foreground=cv2.inRange(hsv,(0,90,65),(22,255,255))
        foreground[crop[:,:,2].astype(float)<crop[:,:,1]*1.5]=0
        foreground=cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        n,components,stats,_=cv2.connectedComponentsWithStats(foreground)
        if n<2:return 'unknown'
        component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        x,y,cw,ch,area=stats[component]
        if area<20 or cw<.5*crop.shape[1] or ch<.5*crop.shape[0]:return 'unknown'
        retained=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=max(5,area*.005))
        retained=retained[retained!=0]
        support=cv2.dilate(np.isin(components,retained).astype(np.uint8),np.ones((5,5),np.uint8))>0
    if np.all(inner[y1:y2,x1:x2][support]>0): return 'safe'
    if np.all(outer[y1:y2,x1:x2][support]==0): return 'not_safe'
    return 'unknown'

@lru_cache(maxsize=2)
def load_model(weights):
    from ultralytics import YOLO
    if not Path(weights).is_file():
        raise FileNotFoundError(f'Trained person_figure weights required: {weights}')
    model=YOLO(weights)
    if model.names != {0:'person_figure'}:
        raise ValueError(f'Expected single person_figure class, got {model.names}')
    return model

def predict_safe_people(image, *, weights=None, zone=None, confidence=0.25, iou=0.45, device='cpu', imgsz=640):
    frame=read_image(image)
    if isinstance(zone,(str,Path)): zone=json.loads(Path(zone).read_text())
    masks=zone_masks(frame,zone)
    model=load_model(str(Path(weights or os.environ.get('FIGURE_WEIGHTS',DEFAULT_WEIGHTS)).resolve()))
    prediction=model.predict(frame,conf=confidence,iou=iou,classes=[0],agnostic_nms=True,
                             imgsz=imgsz,device=device,verbose=False)[0]
    persons=[]
    for box in prediction.boxes:
        bbox=box.xyxy[0].cpu().tolist()
        persons.append(dict(bbox=[round(v,2) for v in bbox],confidence=round(float(box.conf[0]),6),
                            status=judge_safety(bbox,masks,frame.shape,frame=frame)))
    return dict(total_count=len(persons),safe_count=sum(p['status']=='safe' for p in persons),persons=persons)

def count_people(image, **kwargs):
    """Integer figure count, solely from YOLO detections (no filename heuristics)."""
    return predict_safe_people(image, **kwargs)['total_count']

def classify_safety(image, bbox, zone=None):
    """Classify one already-detected figure without loading any YOLO model."""
    frame=read_image(image)
    if isinstance(zone,(str,Path)):zone=json.loads(Path(zone).read_text())
    return judge_safety(bbox,zone_masks(frame,zone),frame.shape,frame=frame)

def draw_result(image,result,zone=None):
    frame=read_image(image).copy()
    if zone and zone.get('polygon'):
        h,w=frame.shape[:2]
        pts=np.rint(np.asarray(zone['polygon'])*[w-1,h-1]).astype(np.int32)
        cv2.polylines(frame,[pts],True,(255,180,0),2)
    colors={'safe':(0,180,0),'not_safe':(0,0,255),'unknown':(0,220,255)}
    for p in result['persons']:
        x1,y1,x2,y2=map(round,p['bbox']); c=colors[p['status']]
        cv2.rectangle(frame,(x1,y1),(x2,y2),c,3)
        cv2.putText(frame,f"{p['status']} {p['confidence']:.2f}",(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,c,2)
    cv2.putText(frame,f"Total: {result['total_count']} Safe: {result['safe_count']}",
                (15,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),4)
    cv2.putText(frame,f"Total: {result['total_count']} Safe: {result['safe_count']}",
                (15,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
    return frame

def main():
    p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--weights',default=str(DEFAULT_WEIGHTS))
    p.add_argument('--zone'); p.add_argument('--output',type=Path,default=ROOT/'predictions')
    p.add_argument('--confidence',type=float,default=.25); p.add_argument('--device',default='cpu')
    args=p.parse_args(); zone=json.loads(Path(args.zone).read_text()) if args.zone else None
    result=predict_safe_people(args.image,weights=args.weights,zone=zone,confidence=args.confidence,device=args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    stem=Path(args.image).stem
    (args.output/f'{stem}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if not cv2.imwrite(str(args.output/f'{stem}.jpg'),draw_result(args.image,result,zone)):
        raise IOError('Could not write result image')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
