"""Behavior checks for safety geometry and image binding; no training metrics."""
import unittest
import numpy as np
from people import zone_masks,judge_safety,image_digest

class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((200,200,3),np.uint8)
        self.zone=dict(reviewed=True,image_size=[200,200],image_sha256=image_digest(self.image),
                       polygon=[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],margin_fraction=.03)
        self.masks=zone_masks(self.image,self.zone)
    def test_inside(self):self.assertEqual(judge_safety([70,70,110,120],self.masks,self.image.shape),'safe')
    def test_outside(self):self.assertEqual(judge_safety([5,60,20,90],self.masks,self.image.shape),'not_safe')
    def test_edge_sitter(self):self.assertEqual(judge_safety([140,70,172,120],self.masks,self.image.shape),'unknown')
    def test_lying_across_boundary(self):self.assertEqual(judge_safety([30,70,120,95],self.masks,self.image.shape),'unknown')
    def test_missing_zone(self):self.assertEqual(judge_safety([70,70,110,120],None,self.image.shape),'unknown')
    def test_truncated(self):self.assertEqual(judge_safety([0,70,20,100],self.masks,self.image.shape),'unknown')
    def test_wrong_image(self):
        image=self.image.copy();image[0,0]=255
        with self.assertRaises(ValueError):zone_masks(image,self.zone)
    def test_unreviewed(self):
        self.zone['reviewed']=False;self.assertIsNone(zone_masks(self.image,self.zone))
    def test_no_color_evidence(self):
        self.assertEqual(judge_safety([70,70,110,120],self.masks,self.image.shape,frame=self.image),'unknown')
    def test_disconnected_limb_is_not_ignored(self):
        image=self.image.copy();image[70:120,100:145]=(20,80,220)
        image[90:105,155:170]=(20,80,220)
        self.assertEqual(judge_safety([97,67,173,123],self.masks,image.shape,frame=image),'unknown')
    def test_crossed_polygon_rejected(self):
        self.zone['polygon']=[[.2,.2],[.8,.8],[.8,.2],[.2,.8]]
        with self.assertRaises(ValueError):zone_masks(self.image,self.zone)

if __name__=='__main__':unittest.main()
