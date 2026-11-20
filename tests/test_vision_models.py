import unittest
import numpy as np
import cv2
import vision_models

class TestVisionModels(unittest.TestCase):
    def test_frame_quality_none(self):
        res = vision_models.analyze_frame_quality(None)
        self.assertFalse(res["valid"])
        self.assertEqual(res["issue"], "No Frame")

    def test_frame_quality_blocked_black(self):
        black_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        res = vision_models.analyze_frame_quality(black_frame)
        self.assertFalse(res["valid"])
        self.assertIn(res["issue"], ("Camera Blocked", "Too Dark", "No Content"))

    def test_frame_quality_overexposed_white(self):
        white_frame = np.ones((224, 224, 3), dtype=np.uint8) * 255
        res = vision_models.analyze_frame_quality(white_frame)
        self.assertFalse(res["valid"])
        self.assertIn(res["issue"], ("Overexposed", "No Content"))

    def test_frame_quality_valid_pattern(self):
        # Create patterned image with edges and variance
        pattern = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 16):
            cv2.line(pattern, (0, i), (224, i), (120, 180, 240), 2)
            cv2.line(pattern, (i, 0), (i, 224), (200, 100, 50), 2)
        res = vision_models.analyze_frame_quality(pattern)
        self.assertTrue(res["valid"])
        self.assertIsNone(res["issue"])

    def test_predict_crop_disease_structure(self):
        # Test with blocked frame
        black_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        res = vision_models.predict_crop_disease(black_frame)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("quality", res)

    def test_predict_presence_structure(self):
        black_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        res = vision_models.predict_presence(black_frame)
        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("quality", res)

if __name__ == '__main__':
    unittest.main()
