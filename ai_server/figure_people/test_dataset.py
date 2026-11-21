"""Guard against duplicate/burst leakage and malformed labels."""
import json
from pathlib import Path
import unittest
from collections import defaultdict

ROOT=Path(__file__).resolve().parent

class DatasetTests(unittest.TestCase):
    def test_no_leakage(self):
        records=json.loads((ROOT/'data/manifest.json').read_text())
        groups=defaultdict(set);hashes=defaultdict(set);counts=defaultdict(int)
        for r in records:
            groups[r['scene_group']].add(r['split']);hashes[r['sha256']].add(r['split'])
            if 'duplicate_of' not in r:counts[r['split']]+=1
        self.assertTrue(all(len(s)==1 for s in groups.values()))
        self.assertTrue(all(len(s)==1 for s in hashes.values()))
        self.assertEqual(dict(counts),{'test':59,'train':215,'val':28})
    def test_exported_labels(self):
        records=json.loads((ROOT/'data/manifest.json').read_text())
        for r in records:
            if 'duplicate_of' in r:continue
            path=ROOT/'data/dataset/labels'/r['split']/f"{r['id']}.txt"
            lines=path.read_text().splitlines();self.assertEqual(len(lines),len(r['boxes']))
            for line in lines:
                cls,cx,cy,w,h=map(float,line.split())
                self.assertEqual(cls,0)
                self.assertTrue(0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1)
                self.assertTrue(cx-w/2>=-1e-6 and cx+w/2<=1+1e-6)
                self.assertTrue(cy-h/2>=-1e-6 and cy+h/2<=1+1e-6)

if __name__=='__main__':unittest.main()
