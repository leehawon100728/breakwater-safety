"""Export visually reviewed scene groups without splitting bursts or duplicates."""
import json,hashlib
from pathlib import Path
from collections import Counter
import shutil

ROOT=Path(__file__).resolve().parent

def group_for(i):
    # Same figure arrangements across non-contiguous sessions stay together.
    groups=[('empty', 'test', [(0,18)]),
      ('one_left','train',[(19,27)]),
      ('one_right_back','train',[(28,42),(54,59),(74,79)]),
      ('one_right_front','val',[(43,53)]),
      ('one_front_outside','test',[(60,73)]),
      ('two_right_deck','train',[(80,115)]),
      ('two_deck_tetra','train',[(116,120),(238,241)]),
      ('two_opposite_edges','train',[(121,128)]),
      ('two_lying_sitting','train',[(129,149)]),
      ('two_left_edge','val',[(150,160)]),
      ('three_spread_edges','val',[(161,166)]),
      ('three_two_left','train',[(167,178)]),
      ('three_two_right','train',[(179,198),(209,221)]),
      ('three_right_line','test',[(199,208),(222,229)]),
      ('three_two_lying','test',[(230,237)]),
      ('three_two_tetra','train',[(242,267)]),
      ('three_opposite_tetra','train',[(268,278)]),
      ('three_lying_tetra','train',[(279,294),(301,305)]),
      ('three_left_tetra','train',[(295,300)])]
    for group,split,ranges in groups:
        if any(a<=i<=b for a,b in ranges): return group,split
    raise ValueError(i)

def main():
    data=ROOT/'data'; records=json.loads((data/'manifest.json').read_text())
    approvals=json.loads((data/'annotation_review.json').read_text())
    seen={}; included=[]; duplicates=[]
    for i,r in enumerate(records):
        fingerprint=hashlib.sha256(json.dumps([r['sha256'],r['boxes']],sort_keys=True).encode()).hexdigest()
        if approvals.get(r['id'])!=fingerprint:
            raise ValueError(f"Unreviewed or changed labels: {r['id']}. Review before export.")
        r['scene_group'],r['split']=group_for(i)
        r['reviewed']=True
        r['review_note']='AI visual review of full-frame contact sheets; color-proposed boxes. Not independent human annotation.'
        if r['sha256'] in seen:
            r['duplicate_of']=seen[r['sha256']];duplicates.append(r['id']);continue
        seen[r['sha256']]=r['id'];included.append(r)
        for kind in ['images','labels']:(data/'dataset'/kind/r['split']).mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/r['image'],data/'dataset/images'/r['split']/f"{r['id']}.jpg")
        w,h=r['size'];lines=[]
        for x1,y1,x2,y2 in r['boxes']:
            lines.append(f'0 {(x1+x2)/2/w:.7f} {(y1+y2)/2/h:.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}')
        (data/'dataset/labels'/r['split']/f"{r['id']}.txt").write_text('\n'.join(lines))
    (data/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    counts=Counter(r['split'] for r in included)
    approval=dict(labels_reviewed=True,scene_groups_reviewed=True,reviewer='AI visual inspection',
                  independent_human_review=False,counts=dict(counts),duplicates_excluded=duplicates,
                  negative_policy='All 19 near-identical empty-scene images held out together for test; no separate empty images in train.',
                  grouping_method='Visual arrangement groups across bursts, merged across repeated sessions; no random frame split.')
    (data/'split_review.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2))
    (data/'dataset.yaml').write_text(f'path: {data / "dataset"}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person_figure\n')
    print(json.dumps(approval,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
