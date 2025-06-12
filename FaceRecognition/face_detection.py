from deepface import DeepFace
import cv2
import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine

cap = cv2.VideoCapture(0)


threshold = 0.3

frame_count = 0

with open("embedding.pkl", 'rb') as f:
    load_dataset = pickle.load(f)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))

    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    try:


        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)


        for idx, face in enumerate(result):
            x , y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            cropped = frame[y:y+h, x:x+w]

            new_face = DeepFace.represent(img_path=cropped, model_name='SFace', enforce_detection = False)


            embedding1 = np.array(new_face[idx]['embedding'])

            min_distance = float('inf')
            identity = "unknown"

            for name, image_vector in load_dataset.items():
                embedding2 = np.array(image_vector)

                similarity = cosine(embedding1, embedding2)

                if similarity < min_distance:
                    min_distance = similarity
                    identity = name
                    
            if min_distance < threshold:
                display_text = f"{identity} ({face['dominant_emotion']})"
            
            else:
                display_text = f"Unknown ({face['dominant_emotion']})"

        
            cv2.putText(frame, display_text, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            
    except Exception as e:
         print(f"No face Detected {e}")

    

    cv2.imshow("DEEPFACE Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()