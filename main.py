import os
import sys
import subprocess
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance
import random

# Auto-install missing packages 
required_packages = ["opencv-contrib-python", "Pillow", "numpy"]
for pkg in required_packages:
    try:
        __import__(pkg if pkg != "opencv-contrib-python" else "cv2")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Paths 
KNOWN_FACES_DIR = "known_faces"
MODELS_DIR = "models"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Clear Old Model 
def clear_old_model():
    for file in ["face_recognizer.yml", "labels.npy"]:
        path = os.path.join(MODELS_DIR, file)
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Removed old model file: {file}")

# Image Augmentation 
def augment_image(img):
    augmented = []
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for _ in range(30):
        temp = pil_img.copy()

        if random.choice([True, False]):
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)

        temp = temp.rotate(random.uniform(-15, 15))

        enhancer = ImageEnhance.Brightness(temp)
        temp = enhancer.enhance(random.uniform(0.7, 1.3))

        temp_cv = cv2.cvtColor(np.array(temp), cv2.COLOR_RGB2BGR)

        if random.choice([True, False]):
            noise = np.random.randint(0, 50, temp_cv.shape, dtype='uint8')
            temp_cv = cv2.add(temp_cv, noise)

        augmented.append(temp_cv)

    return augmented

# Training 
def train_model():
    print("[INFO] Training model with data from", KNOWN_FACES_DIR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_id = 0

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            name = os.path.splitext(filename)[0]
            label_map[current_id] = name

            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(detected_faces) == 0:
                print(f"[WARNING] No face found in {filename}, skipping...")
                continue

            (x, y, w, h) = detected_faces[0]
            face_roi = img[y:y+h, x:x+w]
            augmented_faces = augment_image(face_roi)

            for aug in augmented_faces:
                gray_aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
                faces.append(gray_aug)
                labels.append(current_id)

            current_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(os.path.join(MODELS_DIR, "face_recognizer.yml"))
    np.save(os.path.join(MODELS_DIR, "labels.npy"), label_map)
    print("[INFO] Training complete. Model saved in 'models/'")

# Attendance Logging 
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    with open(ATTENDANCE_FILE, "r") as f:
        lines = f.readlines()
        recorded_names = [line.split(",")[0] for line in lines]

    if name not in recorded_names:
        with open(ATTENDANCE_FILE, "a") as f:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{now}\n")
        print(f"[ATTENDANCE] {name} marked present at {now}")

# Real-time Recognition 
def run_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(MODELS_DIR, "face_recognizer.yml"))
    label_map = np.load(os.path.join(MODELS_DIR, "labels.npy"), allow_pickle=True).item()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    print("[INFO] Starting real-time attendance... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 70:
                name = label_map[face_id]
                mark_attendance(name)
            else:
                name = "Unknown"

            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run 
if __name__ == "__main__":
    clear_old_model()
    train_model()
    run_attendance()