import unittest
import uuid
import main

class TestE2EJourney(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.app.testing = True
        cls.client = main.app.test_client()

    def test_full_farm_management_lifecycle(self):
        """
        Simulate the full user journey:
        1. Farmer visits platform, gets redirected to login.
        2. Signs up with profile.
        3. Logs in to dashboard.
        4. Receives live sensor readings and AI recommendations.
        5. Switches mode to Rule-based and tunes thresholds.
        6. Switches mode to Smart AI mode.
        7. Saves updated farm configuration.
        8. Verifies history logging and logs out cleanly.
        """
        unique = uuid.uuid4().hex[:6]
        email = f"krishi_user_{unique}@agri.in"
        password = "FieldSecret2026"

        # 1. Unauthenticated root access
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 302)

        # 2. Signup
        signup_resp = self.client.post('/signup', data={
            "email": email,
            "password": password,
            "name": "Kisan Bhai",
            "experience": "Expert"
        })
        self.assertEqual(signup_resp.status_code, 302)

        # 3. Login
        login_resp = self.client.post('/login', data={
            "email": email,
            "password": password
        })
        self.assertEqual(login_resp.status_code, 302)

        # 4. View Dashboard
        dash_resp = self.client.get('/')
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn(b"Krishi_Kaar", dash_resp.data)

        # 5. Telemetry & AI recommendations
        sensors_resp = self.client.get('/api/sensors')
        self.assertEqual(sensors_resp.status_code, 200)
        s_data = sensors_resp.get_json()
        self.assertIn("soil_moisture", s_data)
        self.assertIn("water_quality", s_data)

        recs_resp = self.client.get('/api/recommendations')
        self.assertEqual(recs_resp.status_code, 200)
        r_data = recs_resp.get_json()
        self.assertEqual(len(r_data["top_crops"]), 3)
        self.assertIn("fertilizer", r_data)

        # 6. Mode switching: Rule mode with custom thresholds
        rule_resp = self.client.post('/api/control', json={
            "mode": "Rule",
            "rule_thresholds": {
                "moisture_low": 35.0,
                "moisture_high": 65.0,
                "temp_max": 38.0,
                "humidity_min": 25.0
            }
        })
        self.assertEqual(rule_resp.status_code, 200)
        rule_data = rule_resp.get_json()
        self.assertEqual(rule_data["state"]["mode"], "Rule")
        self.assertEqual(rule_data["state"]["rule_thresholds"]["moisture_low"], 35.0)

        # 7. Mode switching: Smart mode
        smart_resp = self.client.post('/api/control', json={"mode": "Smart"})
        self.assertEqual(smart_resp.status_code, 200)
        self.assertEqual(smart_resp.get_json()["state"]["mode"], "Smart")

        # 8. Farm config
        cfg_resp = self.client.post('/api/config', json={
            "farm_area": 8.0,
            "crop_type": "Cotton",
            "soil_type": "Black"
        })
        self.assertEqual(cfg_resp.status_code, 200)
        cfg_data = cfg_resp.get_json()
        self.assertEqual(cfg_data["config"]["crop_type"], "Cotton")

        # 9. Verify history
        hist_resp = self.client.get('/api/history')
        self.assertEqual(hist_resp.status_code, 200)
        self.assertIsInstance(hist_resp.get_json(), list)

        # 10. Clean logout
        logout_resp = self.client.get('/logout')
        self.assertEqual(logout_resp.status_code, 302)

if __name__ == '__main__':
    unittest.main()
