#include <Servo.h>

Servo servo1;
Servo servo2;
int servoPin1 = 6;
int servoPin2 = 7;
void setup() {
  // put your setup code here, to run once:
  servo1.attach(servoPin1);
  servo1.write(100);
  servo2.attach(servoPin2);
  servo2.write(90);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  while(Serial.available() == 0){}
  char incomingByte = Serial.read();
  if (incomingByte == '1'){
    servo1.write(0);
    servo2.write(0);
    delay(5000);
    servo1.write(100);
    servo2.write(90);
  }
  delay(3000);
}
