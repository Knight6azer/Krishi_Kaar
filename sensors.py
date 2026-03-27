import random
import time

# Hardware realignment for Arduino Uno. 
# Raspberry Pi specific libraries (RPi.GPIO, spidev, etc.) are removed.
# This system now supports Soil Sensor, TDS Sensor, DHT11, Ultrasonic, and Salinity.

class SoilMoistureSensor:
    def __init__(self, pin="A0"):
        self.pin = pin

    def read(self):
        # Mocking Arduino analogRead values (0-1023) mapped to percentage
        # Fluctuates around a mean to simulate real-world data better
        return round(random.uniform(20.0, 90.0), 2)

class SoilTemperatureSensor:
    def __init__(self, pin="A3"):
        self.pin = pin

    def read(self):
        # Mocking soil temperature sensor (e.g. DS18B20 inserted in soil)
        return round(random.uniform(18.0, 35.0), 1)

class DHTSensor:
    def __init__(self, pin=2):
        self.pin = pin

    def read(self):
        # Mocking DHT11 digital read
        return round(random.uniform(15.0, 45.0), 1), round(random.uniform(30.0, 95.0), 1)

class TDSSensor:
    def __init__(self, pin="A1"):
        self.pin = pin

    def read(self):
        # Mocking TDS sensor values
        return round(random.uniform(50.0, 2500.0), 1)

class SalinitySensor:
    def __init__(self, pin="A2"):
        self.pin = pin

    def read(self):
        # Mocking EC (Electrical Conductivity) values in mS/cm
        return round(random.uniform(0.1, 5.0), 2)

class UltrasonicSensor:
    def __init__(self, trig_pin=3, echo_pin=4):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

    def read(self):
        # Mock distance: mostly far (> 100cm), rarely close (< 50cm for imposter testing)
        if random.random() < 0.1:
            return round(random.uniform(5.0, 49.0), 2)
        else:
            return round(random.uniform(100.0, 400.0), 2)

# Global instances
soil_sensor = SoilMoistureSensor()
soil_temp_sensor = SoilTemperatureSensor()
dht_sensor = DHTSensor()
tds_sensor = TDSSensor()
salinity_sensor = SalinitySensor()
ultrasonic_sensor = UltrasonicSensor()

def get_all_readings():
    temp, hum = dht_sensor.read()
    return {
        "soil_moisture": soil_sensor.read(),
        "soil_temperature": soil_temp_sensor.read(),
        "air_temperature": temp,
        "humidity": hum,
        "tds": tds_sensor.read(),
        "salinity": salinity_sensor.read(),
        "distance": ultrasonic_sensor.read(),
        "nitrogen": round(random.uniform(10.0, 200.0), 1),
        "phosphorus": round(random.uniform(5.0, 150.0), 1),
        "potassium": round(random.uniform(5.0, 150.0), 1),
        "ph": round(random.uniform(4.0, 9.0), 1)
    }
