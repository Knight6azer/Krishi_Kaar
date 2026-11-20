import unittest
import water_ml

class TestWaterML(unittest.TestCase):
    def test_safe_water(self):
        # Low TDS and normal ambient temp
        pred = water_ml.predict_water_quality(150, 24.0)
        self.assertEqual(pred, "Safe")

    def test_moderate_water(self):
        # Medium TDS
        pred = water_ml.predict_water_quality(650, 28.0)
        self.assertEqual(pred, "Moderate")

    def test_unsafe_water_high_tds(self):
        # High TDS (>1000 ppm)
        pred = water_ml.predict_water_quality(1800, 30.0)
        self.assertEqual(pred, "Unsafe")

    def test_unsafe_water_high_temp(self):
        # Extreme temperature (>40°C)
        pred = water_ml.predict_water_quality(250, 48.0)
        self.assertEqual(pred, "Unsafe")

    def test_input_resilience(self):
        # Edge cases: negative, extreme values, strings, None
        self.assertIn(water_ml.predict_water_quality(-50, 25), ("Safe", "Moderate", "Unsafe"))
        self.assertIn(water_ml.predict_water_quality(3000, 80), ("Safe", "Moderate", "Unsafe"))
        self.assertIn(water_ml.predict_water_quality("350", "25"), ("Safe", "Moderate", "Unsafe"))
        self.assertIn(water_ml.predict_water_quality(None, None), ("Safe", "Moderate", "Unsafe"))

if __name__ == '__main__':
    unittest.main()
