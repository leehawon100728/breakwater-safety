"""Convert HEIC originals and propose boxes. Proposals require visual review."""
from pathlib import Path
import argparse, hashlib, json, unicodedata
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import pillow_heif

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    args = ap.parse_args()
    dest = ROOT / 'data'
    (dest / 'images').mkdir(parents=True, exist_ok=True)
    (dest / 'review').mkdir(exist_ok=True)
    records = []
    for folder in sorted(args.source.iterdir()):
        if not folder.is_dir(): continue
        category = unicodedata.normalize('NFC', folder.name)
        for src in sorted(folder.iterdir()):
            if src.suffix.lower() not in {'.heic', '.jpg', '.jpeg', '.png'}: continue
            key = f'{len(records):03d}_{src.stem}'
            im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
            original_size = list(im.size)
            im.thumbnail((1280,1280))
            out = dest / 'images' / f'{key}.jpg'
            im.save(out, quality=95)
            a = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(a, (0,140,110),(12,255,255))
            rgb=np.array(im)
            mask[rgb[:,:,0].astype(float)<rgb[:,:,1]*1.8]=0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            boxes=[]
            for x,y,w,h,area in stats[1:]:
                if area < 90 or w < 6 or h < 8: continue
                pad=4
                boxes.append([max(0,int(x)-pad),max(0,int(y)-pad),min(im.width,int(x+w)+pad),min(im.height,int(y+h)+pad)])
            thumb=np.array(im.resize((32,32)).convert('L'))
            records.append(dict(id=key,source=str(src),category=category,image=str(out.relative_to(ROOT)),
                original_size=original_size,size=list(im.size),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                boxes=boxes,reviewed=False,scene_group=None,thumb=thumb.flatten().tolist()))
            print(key,len(boxes),flush=True)
    (dest/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    for start in range(0,len(records),20):
        sheet=Image.new('RGB',(1200,1500),'#dddddd'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(records[start:start+20]):
            im=Image.open(ROOT/r['image']); d=ImageDraw.Draw(im)
            for k,b in enumerate(r['boxes']):
                d.rectangle(b,outline='red',width=4); d.text((b[0],b[1]),str(k),fill='black')
            im.thumbnail((290,265))
            x=(j%4)*300; y=(j//4)*300
            sheet.paste(im,(x,y+25)); draw.text((x+4,y+5),f"{r['id']} n={len(r['boxes'])}",fill='black')
        sheet.save(dest/'review'/f'boxes_{start:03d}.jpg',quality=92)

if __name__=='__main__': main()
