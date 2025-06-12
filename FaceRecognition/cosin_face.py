import pickle 
import os
from deepface import DeepFace
import cv2
import numpy as np

with open('embedding.pkl', 'rb') as f:
    data = pickle.load(f)


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    if not ret:
       break

    
    new_face = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)[0]['embedding']
    result = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)

    for face in result:

        x,y,w,h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        cv2.rectangle(frame, (x,y), (x+w , y+h), (255,0,0),2)

        for name,data_faces in data.items():
            embedding1 = np.array(new_face)
            embedding2 = np.array(data_faces)

            cosine_sim = np.dot(embedding1 , embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            similarity = 1- cosine_sim

            if similarity < min_distance:
                min_distance = similarity
                identity = name

        text = f"{face['age']}"
        cv2.putText(frame, text,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

        cv2.imshow("hello", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
threshold = 0.3

if min_distance < threshold:
    print(f"face matched with {identity} with distance {min_distance}")

else:
    print("Unknown Face")

cap.release()
cv2.destroyAllWindow()
        