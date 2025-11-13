#include <Arduino.h>
int ledPin = 13;          // LED connected to pin 13
int dotTime = 200;        // Duration of a dot in milliseconds
String inputMessage = ""; // Stores the message from Serial Monitor

// Function to return Morse code for each letter
String morseChar(char c) {
  switch (toupper(c)) {
    case 'A': return ".-";
    case 'B': return "-...";
    case 'C': return "-.-.";
    case 'D': return "-..";
    case 'E': return ".";
    case 'F': return "..-.";
    case 'G': return "--.";
    case 'H': return "....";
    case 'I': return "..";
    case 'J': return ".---";
    case 'K': return "-.-";
    case 'L': return ".-..";
    case 'M': return "--";
    case 'N': return "-.";
    case 'O': return "---";
    case 'P': return ".--.";
    case 'Q': return "--.-";
    case 'R': return ".-.";
    case 'S': return "...";
    case 'T': return "-";
    case 'U': return "..-";
    case 'V': return "...-";
    case 'W': return ".--";
    case 'X': return "-..-";
    case 'Y': return "-.--";
    case 'Z': return "--..";
    case '0': return "-----";
    case '1': return ".----";
    case '2': return "..---";
    case '3': return "...--";
    case '4': return "....-";
    case '5': return ".....";
    case '6': return "-....";
    case '7': return "--...";
    case '8': return "---..";
    case '9': return "----.";
    case ' ': return " ";  // Space between words
    default: return "";    // Ignore unknown characters
  }
}

void dot() {
  digitalWrite(ledPin, HIGH);
  delay(dotTime);
  digitalWrite(ledPin, LOW);
  delay(dotTime);
}

void dash() {
  digitalWrite(ledPin, HIGH);
  delay(dotTime * 3);
  digitalWrite(ledPin, LOW);
  delay(dotTime);
}

void morseLetter(String code) {
  for (int i = 0; i < code.length(); i++) {
    if (code[i] == '.') dot();
    else if (code[i] == '-') dash();
  }
  delay(dotTime * 3); // Space between letters
}

void sendMessage(String message) {
  for (int i = 0; i < message.length(); i++) {
    if (message[i] == ' ') {
      delay(dotTime * 7); // Word gap
    } else {
      String code = morseChar(message[i]);
      morseLetter(code);
    }
  }
}

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("Enter a message to blink in Morse code:");
}

void loop() {
  if (Serial.available() > 0) {
    inputMessage = Serial.readStringUntil('\n');  // Read input line
    inputMessage.trim(); // Remove extra spaces
    Serial.print("Blinking message: ");
    Serial.println(inputMessage);
    sendMessage(inputMessage);
    Serial.println("Done! Enter another message:");
  }
}
