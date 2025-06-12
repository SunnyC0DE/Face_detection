import os
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pickle
from image_vectorization import vector

cap = cv2.VideoCapture(0)


image_path = "Project Folder/unknown"

frame_count = 0
while True:

    ret, frame = cap.read()

    frame = cv2.resize(frame, (320, 240))

    if not ret:
        break

    text = ""

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    try:

        result = DeepFace.analyze(frame, actions=['age'] ,enforce_detection=False)


        min_distance = float("inf")
        identity = "unknown"

        for idx, face in enumerate(result):
            x,y,w,h = face['region']['x'],face['region']['y'],face['region']['w'],face['region']['h']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)

            cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2 )


    except Exception as e:
        print(f" no face detected: {e}")

    cv2.imshow("Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):

        new_path =  os.path.join(image_path, "unknown.jpg")

        cv2.imwrite(new_path, frame)
        vector(image_path)
        break

    elif key == ord('q'):
        break

    else:
        continue

cap.release()
cv2.destroyAllWindows()
