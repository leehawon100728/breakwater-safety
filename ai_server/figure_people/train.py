"""Train only after reviewed labels and scene groups have been exported."""
from pathlib import Path
import argparse
import json
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='yolo11n.pt')
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=8); p.add_argument('--device',default='auto')
    args=p.parse_args()
    approval=json.loads((ROOT/'data'/'split_review.json').read_text())
    if not approval.get('labels_reviewed') or not approval.get('scene_groups_reviewed'):
        raise ValueError('Review all boxes and scene groups before training')
    device=args.device
    if device=='auto':
        device=0 if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Training device:',device)
    YOLO(args.model).train(data=str(ROOT/'data'/'dataset.yaml'),epochs=args.epochs,
        imgsz=args.imgsz,batch=args.batch,device=device,project=str(ROOT/'runs'),name='figure_detector',
        exist_ok=False,seed=42,workers=0,patience=20,save=True,plots=True,
        degrees=10,translate=.05,scale=.15,fliplr=.5,flipud=0,mosaic=0,mixup=0,
        hsv_h=.01,hsv_s=.15,hsv_v=.2)

if __name__=='__main__': main()
