import random
import time

# Try to import hardware libraries, use mock if not available (for development on PC)
try:
    import RPi.GPIO as GPIO
    import Adafruit_DHT
    import spidev
    from w1thermsensor import W1ThermSensor
    IS_RPI = True
except ImportError:
    IS_RPI = False
    print("Running in MOCK mode (Hardware libraries not found)")

class SoilMoistureSensor:
    def __init__(self, channel=0):
        self.channel = channel
        if IS_RPI:
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 1350000

    def read(self):
        if IS_RPI:
            adc = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
            data = ((adc[1] & 3) << 8) + adc[2]
            # Map 0-1023 to 0-100% (approximate, adjust based on calibration)
            percentage = max(0, min(100, (1023 - data) / 1023 * 100))
            return round(percentage, 2)
        else:
            return round(random.uniform(30.0, 80.0), 2)  # Mock value

class DHTSensor:
    def __init__(self, pin=4):
        self.pin = pin
        self.sensor = Adafruit_DHT.DHT11 if IS_RPI else None

    def read(self):
        if IS_RPI:
            humidity, temperature = Adafruit_DHT.read_retry(self.sensor, self.pin)
            return temperature, humidity
        else:
            return round(random.uniform(20.0, 35.0), 1), round(random.uniform(40.0, 70.0), 1)

class TDSSensor:
    def __init__(self, channel=1):
        self.channel = channel
        if IS_RPI:
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 1350000

    def read(self):
        if IS_RPI:
            adc = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
            data = ((adc[1] & 3) << 8) + adc[2]
            voltage = (data * 3.3) / 1024.0
            tds_value = (133.42 * voltage * voltage * voltage - 255.86 * voltage * voltage + 857.39 * voltage) * 0.5 
            return round(tds_value, 2)
        else:
            return round(random.uniform(100.0, 600.0), 1)

class DS18B20Sensor:
    def __init__(self):
        if IS_RPI:
            self.sensor = W1ThermSensor()

    def read(self):
        if IS_RPI:
            return self.sensor.get_temperature()
        else:
            return round(random.uniform(20.0, 30.0), 2)

# Global instances
soil_sensor = SoilMoistureSensor()
dht_sensor = DHTSensor()
tds_sensor = TDSSensor()
temp_sensor = DS18B20Sensor()

def get_all_readings():
    temp, hum = dht_sensor.read()
    return {
        "soil_moisture": soil_sensor.read(),
        "air_temperature": temp,
        "humidity": hum,
        "tds": tds_sensor.read(),
        "water_temperature": temp_sensor.read()
    }
