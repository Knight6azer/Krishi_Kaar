// Arduino Code for Smart Agriculture Project - Responsive Edition
// This code uses non-blocking timing to ensure instant command response.

const int ledPin = LED_BUILTIN;
unsigned long lastSendTime = 0;
const long interval = 2000; // Send sensor data every 2 seconds

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
}

void loop() {
  // 1. Process Commands INSTANTLY (No delay anymore!)
  while (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') {
      digitalWrite(ledPin, HIGH);
    } else if (command == '0') {
      digitalWrite(ledPin, LOW);
    }
  }

  // 2. Broadcast Sensor Data on a Non-blocking Timer
  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= interval) {
    lastSendTime = currentTime;

    // Generate/Read Sensors
    int soil = random(400, 900);
    float temp = 24.0 + (random(0, 50) / 10.0);
    float hum = 50.0 + (random(0, 100) / 10.0);

    // Send to Python
    Serial.print(soil);
    Serial.print(",");
    Serial.print(temp);
    Serial.print(",");
    Serial.println(hum);
  }
}
