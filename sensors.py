import random
import time

# Hardware realignment for Arduino Uno. 
# Raspberry Pi specific libraries (RPi.GPIO, spidev, etc.) are removed.
# This system now supports Soil Sensor, TDS Sensor, DHT11, and Ultrasonic.

class SoilMoistureSensor:
    def __init__(self, pin="A0"):
        self.pin = pin

    def read(self):
        # Mocking Arduino analogRead values (0-1023) mapped to percentage
        return round(random.uniform(30.0, 80.0), 2)

class DHTSensor:
    def __init__(self, pin=2):
        self.pin = pin

    def read(self):
        # Mocking DHT11 digital read
        return round(random.uniform(20.0, 35.0), 1), round(random.uniform(40.0, 70.0), 1)

class TDSSensor:
    def __init__(self, pin="A1"):
        self.pin = pin

    def read(self):
        # Mocking TDS sensor values
        return round(random.uniform(100.0, 600.0), 1)

class UltrasonicSensor:
    def __init__(self, trig_pin=3, echo_pin=4):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

    def read(self):
        # Mock distance: mostly far (> 100cm), rarely close (< 50cm for imposter testing)
        if random.random() < 0.1:
            return round(random.uniform(10.0, 45.0), 2)
        else:
            return round(random.uniform(100.0, 300.0), 2)

# Global instances
soil_sensor = SoilMoistureSensor()
dht_sensor = DHTSensor()
tds_sensor = TDSSensor()
ultrasonic_sensor = UltrasonicSensor()

def get_all_readings():
    temp, hum = dht_sensor.read()
    return {
        "soil_moisture": soil_sensor.read(),
        "air_temperature": temp,
        "humidity": hum,
        "tds": tds_sensor.read(),
        "distance": ultrasonic_sensor.read(),
        "nitrogen": round(random.uniform(20.0, 150.0), 1),
        "phosphorus": round(random.uniform(10.0, 100.0), 1),
        "potassium": round(random.uniform(10.0, 100.0), 1),
        "ph": round(random.uniform(5.5, 8.5), 1)
    }
