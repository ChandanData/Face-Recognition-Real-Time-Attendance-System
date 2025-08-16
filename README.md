# Face-Recognition-Real-Time-Attendance-System
Detect face in real-time through saved faces and marks attendance in a CSV file.
A **Python-based face recognition attendance system** built with **OpenCV, NumPy, and Pillow** that performs **real-time face detection, recognition, and automatic attendance logging** into a CSV file.
# Features
- Real-time face detection & recognition via webcam
- Automatic attendance logging with name, date, and time
- Image augmentation for improved model accuracy
- Easy to extend by adding images to the `known_faces/` folder
# Requirements
- Python 3.8+
- Packages (see `requirements.txt`):
  - opencv-contrib-python  
  - numpy  
  - Pillow  
Install all dependencies with:
```bash
pip install -r requirements.txt

# Run
1. Start the attendance system:
   ```bash
   python main.py
# Quit Guide
Press q on your keyboard while the webcam window is open.
The webcam will close and the program will stop 
