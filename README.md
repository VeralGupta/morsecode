# 📡 Optical Morse Code Communication System

### **Android → Arduino → LED → Laptop Webcam → Python Decoder**

A complete wireless optical communication system using Morse Code blinks.

---

## 📘 **Project Overview**

This project creates a **wireless optical link** between an **Android phone** and a **Laptop** using:

* Android phone compiles & uploads code to Arduino
* Arduino Uno blinks an LED in Morse Code
* Laptop webcam detects LED blink timings
* Python + OpenCV decodes Morse → text in real-time

This all works **without Bluetooth, WiFi, or radio** — only **light**.

---

## ⚙️ **System Architecture**

```
Android Phone (Termux + PlatformIO)
            ↓ USB-OTG
Arduino Uno (LED Morse Transmitter)
            ↓ Light Blinks
Laptop Webcam (Real-time detection)
            ↓ OpenCV
Python Decoder → Output Text
```

---

# 🧩 **Features**

### ✔ Arduino Morse transmitter

* Sends *any text* typed in Serial
* Automatic DOT/DASH timing
* Standard Morse code mapping
* Constant timing for receiver synchronization

### ✔ Python webcam receiver

* Automatic LED detection (Green color filtering)
* Real-time ON/OFF duration measurement
* Auto-calibration using SOS or user-given word
* Full decoding of Morse into text
* Live UI with camera preview, brightness, FPS
* Supports recalibration, message reset, word gaps

### ✔ Android workflows

* Entire project built and flashed via **Termux + PlatformIO**
* No PC needed to program Arduino
* No ArduinoDroid required

---

# 🧱 **Hardware Required**

| Component                                | Purpose                    |
| ---------------------------------------- | -------------------------- |
| **Arduino Uno**                          | Morse transmitter          |
| **Green LED + 220Ω resistor**            | Visual signaling           |
| **Breadboard + Jumper wires**            | Connections                |
| **Android Phone (OTG supported)**        | Compiling & flashing code  |
| **USB-OTG adapter (Type-C to USB-A)**    | Phone ↔ Arduino connection |
| **Laptop with Webcam**                   | Optical receiver           |
| **Ambient lighting (normal room light)** | Improving detection        |

---

# 🧩 **Software Required**

### **On Android:**

* **Termux** (from F-Droid)
* `termux-api` (optional)
* Debian (via proot)
* Python3 + venv
* PlatformIO (`pip install platformio`)
* Stable OTG + USB permissions

### **On Laptop:**

* Python 3.8+
* OpenCV
* Numpy
* Pillow (PIL)
* util.py (color threshold helper)

Install dependencies:

```bash
pip install opencv-python numpy pillow
```

---

# 🚀 **1. Arduino Transmitter Setup (Android Phone)**

### **1️⃣ Install Termux**

(From F-Droid only — Play Store is outdated)

### **2️⃣ Install Debian**

```bash
pkg install proot-distro
proot-distro install debian
proot-distro login debian
```

### **3️⃣ Install PlatformIO inside Debian**

```bash
apt update
apt install python3 python3-venv -y
python3 -m venv ~/pioenv
source ~/pioenv/bin/activate

pip install platformio
```

### **4️⃣ Create Project**

```bash
pio project init --project-dir ~/morse_tx --board uno
```

### **5️⃣ Paste transmitter code into `src/main.cpp`**

(Your Morse transmitter code)

### **6️⃣ Build**

```bash
cd ~/morse_tx
pio run
```

### **7️⃣ Upload to Arduino**

On Redmi phones (OTG supported):

```bash
pio run -t upload --upload-port /dev/ttyACM0
```

---

# 💡 **Arduino Transmitter Code Features**

* Interactive input via Serial
* LED on Pin 13 blinks DOT (200ms) and DASH (600ms)
* Automatically handles letter + word spacing
* Compatible with camera-based decoding

---

# 🚀 **2. Laptop Receiver Setup (Python + OpenCV)**

### **Clone repo**

```bash
git clone https://github.com/yourusername/optical-morse-comm
cd optical-morse-comm
```

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Run the receiver**

```bash
python receiver.py
```

---

# 🧠 **Receiver Functionality**

* Detects green LED using HSV mask
* Measures ON/OFF times with ms accuracy
* Distinguishes DOT vs DASH using calibration
* Handles:

  * letter gaps
  * word gaps
  * debouncing
* Provides real-time UI:

  * Camera feed
  * Detected bounding box
  * Current symbol
  * Decoded message
  * FPS indicator
  * Calibration prompts

---

# 🔧 **Calibration Process**

On first run, program requests a calibration word:

Example:

```
Enter calibration word (example: SOS):
```

Then:

* Flash the LED sending that word repeatedly
* System collects DOT and DASH timings
* Builds dynamic thresholds:

  * dot/dash cutoff
  * letter gap
  * word gap

Saved as:

```
calibration.json
```

Calibration works even in noisy lighting.

---

# 📝 **Usage Workflow**

## **Transmitter**

1. Upload code to Arduino
2. Open Serial Monitor
3. Type:

```
HELLO WORLD
```

4. LED blinks Morse code

## **Receiver**

1. Run Python script
2. Point webcam at LED (20–40 cm away)
3. See decoded message live
4. Press:

   * **q** → quit
   * **r** → reset message
   * **c** → recalibrate
   * **f** → toggle FPS

---

# 📊 File Structure

```
optical-morse-comm/
│
├── transmitter/          
│   └── main.cpp          # Arduino code
│
├── receiver/
│   ├── receiver.py       # Python OpenCV decoder (YOUR CODE)
│   ├── util.py           # HSV threshold helper
│   ├── calibration.json  # Auto-generated
│
├── README.md
├── requirements.txt
```

---

# 🛠 Troubleshooting

### ❌ Webcam not detected

Use:

```bash
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### ❌ Too many wrong dots/dashes

→ Re-run calibration (`c`)

### ❌ LED not detected

Adjust LED brightness, avoid direct sunlight.

### ❌ Termux cannot see Arduino

Check OTG cable / settings / permissions.

---
* Robust receiver logic
ack"**.
