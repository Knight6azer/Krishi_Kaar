import unittest
import sensors

class TestSensors(unittest.TestCase):
    def test_smoothed_sensor_bounds(self):
        sensor = sensors.SmoothedSensor(initial=50.0, min_val=10.0, max_val=90.0, max_delta=2.0)
        for _ in range(100):
            val = sensor.read()
            self.assertGreaterEqual(val, 10.0)
            self.assertLessEqual(val, 90.0)

    def test_smoothed_sensor_override(self):
        sensor = sensors.SmoothedSensor(initial=50.0, min_val=0.0, max_val=100.0, max_delta=1.0)
        sensor.set(85.5)
        val = sensor.read()
        self.assertAlmostEqual(val, 85.5, delta=2.0)

    def test_get_all_readings_structure(self):
        readings = sensors.get_all_readings()
        expected_keys = {
            "soil_moisture", "soil_temperature", "air_temperature",
            "humidity", "tds", "salinity", "distance",
            "nitrogen", "phosphorus", "potassium", "ph", "source"
        }
        self.assertTrue(expected_keys.issubset(readings.keys()))
        self.assertIn(readings["source"], ("arduino", "simulation"))
        self.assertGreaterEqual(readings["soil_moisture"], 0)
        self.assertLessEqual(readings["soil_moisture"], 100)
        self.assertGreaterEqual(readings["ph"], 0)
        self.assertLessEqual(readings["ph"], 14)

    def test_arduino_connection_and_pump_command_graceful(self):
        self.assertIsInstance(sensors.is_arduino_connected(), bool)
        # When no hardware is connected, sending command returns False without exception
        res = sensors.send_pump_command("ON")
        self.assertIsInstance(res, bool)

if __name__ == '__main__':
    unittest.main()
