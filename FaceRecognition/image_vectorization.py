import os
import shutil
import pickle
from deepface import DeepFace
import numpy as np

class vector:
    def __init__(self,path):
        self.path = path
        self.comparing_path = "Project Folder/faces"
        self.embedding_file = "embedding.pkl"
        self.load_embeddings()
        self.checking()

    def load_embeddings(self):
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                self.embeddings = pickle.load(f)

        else:
            self.embeddings = {}

    def save_embeddings(self):
        with open(self.embedding_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def checking(self):
        unknown_images = os.listdir(self.path)

        if not unknown_images:
            return  
        
        unknown_image_path = os.path.join(self.path, unknown_images[0])
        
        for image_name in os.listdir(self.comparing_path):
            registered_image_path = os.path.join(self.comparing_path, image_name)

            try :
                result = DeepFace.verify(registered_image_path, unknown_images, model_name="SFace")

                if result['verified']:
                    os.remove(unknown_image_path)
                    return
            
            except Exception as e:
                os.remove(unknown_image_path)
                return
            
        print("New Face Detected! Processing to registration")

        try:

            embedding_obj = DeepFace.represent(unknown_image_path, model_name="SFace", enforce_detection=False)

            if embedding_obj:
                embedding = np.array(embedding_obj[0]['embedding'])

            else:
                os.remove(unknown_image_path)
                return
            
            name = input("enter your Name: ").strip()

            if not name:
                os.remove(unknown_image_path)
                return
            
            new_filename = f"{name}.jpg"
            new_path = os.path.join(self.comparing_path, new_filename)
            shutil.move(unknown_image_path, new_path)

            self.embeddings[name] = embedding
            self.save_embeddings()

            print(f"Registration Sucessful! Welcome, {name}.")

        except:
            os.remove(unknown_image_path)