"""Create an image-specific ROI by clicking the deck boundary; run locally."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from people import read_image,image_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('image');ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();image=read_image(args.image);h,w=image.shape[:2]
    scale=min(1100/w,800/h,1);preview=cv2.resize(image,None,fx=scale,fy=scale)
    points=[]
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:points.append((x,y))
        elif event==cv2.EVENT_RBUTTONDOWN and points:points.pop()
    cv2.namedWindow('Deck ROI');cv2.setMouseCallback('Deck ROI',click)
    print('Click only the flat deck top boundary in order. Right click: undo. Enter: save. Esc: cancel.')
    while True:
        frame=preview.copy()
        if points:cv2.polylines(frame,[np.array(points,np.int32)],len(points)>2,(255,180,0),2)
        cv2.imshow('Deck ROI',frame);k=cv2.waitKey(30)&255
        if k==27:cv2.destroyAllWindows();return
        if k in (10,13) and len(points)>=3:break
    zone=dict(image_size=[w,h],image_sha256=image_digest(image),
        polygon=[[min(1,x/scale/(w-1)),min(1,y/scale/(h-1))] for x,y in points],
        margin_fraction=.02,reviewed=True,annotation_source='user clicked deck boundary')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(zone,indent=2));cv2.destroyAllWindows()

if __name__=='__main__':main()
