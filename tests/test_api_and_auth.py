import unittest
import json
import uuid
import main

class TestAPIAndAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.app.testing = True
        cls.client = main.app.test_client()

    def setUp(self):
        self.unique_id = uuid.uuid4().hex[:8]
        self.test_email = f"farmer_{self.unique_id}@krishi.test"
        self.test_password = "SecurePassword123"

    def test_public_routes(self):
        # /auth should be accessible
        res = self.client.get('/auth')
        self.assertEqual(res.status_code, 200)

        # /api/translations/en should be accessible without auth
        res_trans = self.client.get('/api/translations/en')
        self.assertEqual(res_trans.status_code, 200)
        data = res_trans.get_json()
        self.assertIn("dashboard", data)

        # /api/system_status should be accessible without auth
        res_sys = self.client.get('/api/system_status')
        self.assertEqual(res_sys.status_code, 200)
        sys_data = res_sys.get_json()
        self.assertEqual(sys_data["status"], "running")

    def test_unauthorized_redirect_or_401(self):
        # Dashboard page without login redirects to /auth
        res_index = self.client.get('/')
        self.assertEqual(res_index.status_code, 302)
        self.assertIn('/auth', res_index.headers.get('Location', ''))

        # Protected API endpoints without login return 401
        for ep in ['/api/sensors', '/api/recommendations', '/api/vision', '/api/history']:
            res = self.client.get(ep)
            self.assertEqual(res.status_code, 401, f"Expected 401 on {ep}, got {res.status_code}")
            data = res.get_json()
            self.assertIn("error", data)

    def test_signup_validation(self):
        # Invalid email
        res_bad_email = self.client.post('/signup', data={
            "email": "notanemail",
            "password": "validpassword",
            "name": "Test"
        })
        self.assertEqual(res_bad_email.status_code, 302)
        self.assertIn('error', res_bad_email.headers.get('Location', ''))

        # Too short password (<6 chars)
        res_short_pw = self.client.post('/signup', data={
            "email": f"valid_{self.unique_id}@test.com",
            "password": "123",
            "name": "Test"
        })
        self.assertEqual(res_short_pw.status_code, 302)
        self.assertIn('error', res_short_pw.headers.get('Location', ''))

    def test_signup_login_lifecycle(self):
        # 1. Signup
        res_signup = self.client.post('/signup', data={
            "email": self.test_email,
            "password": self.test_password,
            "name": "Green Farmer",
            "experience": "Advanced"
        })
        self.assertEqual(res_signup.status_code, 302)
        self.assertIn('success', res_signup.headers.get('Location', ''))

        # 2. Login
        res_login = self.client.post('/login', data={
            "email": self.test_email,
            "password": self.test_password
        })
        self.assertEqual(res_login.status_code, 302)
        self.assertEqual(res_login.headers.get('Location'), '/')

        # 3. Access authenticated endpoints with session
        res_sensors = self.client.get('/api/sensors')
        self.assertEqual(res_sensors.status_code, 200)
        s_data = res_sensors.get_json()
        self.assertIn('soil_moisture', s_data)

        res_recs = self.client.get('/api/recommendations')
        self.assertEqual(res_recs.status_code, 200)
        r_data = res_recs.get_json()
        self.assertIn('top_crops', r_data)

        # 4. Mode control
        res_ctrl = self.client.post('/api/control', json={"mode": "Manual", "pump": "ON"})
        self.assertEqual(res_ctrl.status_code, 200)
        c_data = res_ctrl.get_json()
        self.assertEqual(c_data["state"]["pump"], "ON")

        # 5. Config update
        res_cfg = self.client.post('/api/config', json={"farm_area": 12.5, "crop_type": "Rice"})
        self.assertEqual(res_cfg.status_code, 200)
        cfg_data = res_cfg.get_json()
        self.assertEqual(cfg_data["config"]["farm_area"], 12.5)

        # 6. Logout
        res_logout = self.client.get('/logout')
        self.assertEqual(res_logout.status_code, 302)

        # 7. Post-logout protected access returns 401
        res_post_logout = self.client.get('/api/sensors')
        self.assertEqual(res_post_logout.status_code, 401)

if __name__ == '__main__':
    unittest.main()
