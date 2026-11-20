import unittest
import agri_ai

class TestAgriAI(unittest.TestCase):
    def setUp(self):
        self.normal_readings = {
            "nitrogen": 80,
            "phosphorus": 45,
            "potassium": 50,
            "air_temperature": 26,
            "humidity": 65,
            "ph": 6.8,
            "soil_moisture": 55,
            "salinity": 1.5,
            "rainfall": 120
        }

    def test_recommendations_keys(self):
        recs = agri_ai.get_recommendations(self.normal_readings)
        expected_keys = {
            "top_crops", "top_crops_detailed", "crop",
            "fertilizer", "irrigation", "irrigation_code",
            "irrigation_label", "health_score"
        }
        self.assertTrue(expected_keys.issubset(recs.keys()))
        self.assertEqual(len(recs["top_crops"]), 3)
        self.assertEqual(len(recs["top_crops_detailed"]), 3)
        self.assertIn(recs["irrigation"], ("ON", "OFF"))
        self.assertIn(recs["irrigation_code"], (0, 1))

    def test_top_crops_confidence_ordered(self):
        recs = agri_ai.get_recommendations(self.normal_readings)
        confidences = [c["confidence"] for c in recs["top_crops_detailed"]]
        self.assertGreaterEqual(confidences[0], confidences[1])
        self.assertGreaterEqual(confidences[1], confidences[2])

    def test_health_score_range(self):
        recs = agri_ai.get_recommendations(self.normal_readings)
        self.assertGreaterEqual(recs["health_score"], 5)
        self.assertLessEqual(recs["health_score"], 100)

    def test_out_of_bounds_resilience(self):
        extreme_readings = {
            "nitrogen": -100,
            "phosphorus": 9999,
            "potassium": None,
            "air_temperature": 150,
            "humidity": -20,
            "ph": 25,
            "soil_moisture": -50,
            "salinity": 100
        }
        recs = agri_ai.get_recommendations(extreme_readings)
        self.assertIn("crop", recs)
        self.assertIn("health_score", recs)
        self.assertGreaterEqual(recs["health_score"], 5)
        self.assertLessEqual(recs["health_score"], 100)

    def test_empty_dict_fallback(self):
        recs = agri_ai.get_recommendations({})
        self.assertIn("crop", recs)
        self.assertIn("health_score", recs)

if __name__ == '__main__':
    unittest.main()
