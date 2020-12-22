char blueToothVal;
 
// Motor A connections
const int enA = 9;
const int in1 = 8;
const int in2 = 7;

// Motor B connections
const int enB = 3;
const int in3 = 5;
const int in4 = 4;

// Speed Control
int speedControl = 0;

void setup(){

  // bluetooth connection with raspberrypi 
  Serial.begin(9600);
  
  // Set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  // Turn off motors - Initial state
  brakes();
}

void loop(){

  // Speed Control
  analogWrite(enA, 60);
  analogWrite(enB, 60); 

  // Check for bluetooth input from raspberry pi
  if(Serial.available()){
    blueToothVal = Serial.read();
  }

  if (blueToothVal==' '){
    brakes();
    blueToothVal = '-';
  }

  if (blueToothVal=='w'){
    forward();
    delay(50);
    brakes();
    blueToothVal = '-';
  }
  
  else if (blueToothVal=='a'){
    left();
    delay(100);
    brakes();
    blueToothVal = '-';
  }

  else if (blueToothVal=='s'){
    reverse();
    delay(100);
    brakes();
    blueToothVal = '-';
  }
  
  else if (blueToothVal=='d'){
    right();
    delay(100);
    brakes();
    blueToothVal = '-';
  }
  
}

void forward(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);  
}

void reverse(){
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void left(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void right(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void brakes(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}
