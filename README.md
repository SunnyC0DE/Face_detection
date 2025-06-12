# 😎 Face Detection & Recognition System (DeepFace)

A real-time face detection and recognition system built using Python and DeepFace. The system captures faces from a webcam, compares them with known images, and displays or logs the result.

---

## 📌 Description

This project uses the DeepFace library to detect and recognize faces in real-time using a webcam. It leverages deep learning models (like VGG-Face, Facenet, etc.) for high-accuracy face recognition.

You can use it to:
- Detect faces live from webcam
- Recognize known people from a stored database
- Compare face similarity
- Optionally record attendance (extensible)

---

## 🧠 Features

- Real-time webcam feed using OpenCV
- Face detection and recognition using DeepFace
- Match faces with stored dataset
- Displays matched name with confidence score
- Optionally extensible to mark attendance/logs

---

## 📂 Project Structure

# FaceRecoginition

d-----         Project Folder
d-----         __pycache__
-a----         cosin_face.py
-a----         embedding.pkl
-a----         face_detection.py
-a----         face_register.py
-a----         image_vectorization.py



---

## 🔧 Technologies Used

- Python 3
- DeepFace
- OpenCV (`cv2`)
- NumPy
- PIL (optional for image handling)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/SunnyC0DE/face-detection.git
cd face-recognition-deepface

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
