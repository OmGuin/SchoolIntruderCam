import os
import face_recognition
database_path = r"C:\Users\omgui\Downloads\sciencefair\sciencefair\sample_database"
known_face_encodings = []
known_face_names = []



def load_known_faces():
    for filename in os.listdir(database_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(database_path, filename)
            # Load an image
            image = face_recognition.load_image_file(image_path)
            # Get the face encoding for the loaded image
            face_encoding = face_recognition.face_encodings(image)[0]
            # Append to our known face encodings and names
            known_face_encodings.append(face_encoding)
            known_face_names.append(filename.split('.')[0])  # Name is the filename without extension
load_known_faces()

if __name__ == "__main__":
    #print(known_face_encodings)
    
    print("Database setup for: ")
    print("\n".join(known_face_names))