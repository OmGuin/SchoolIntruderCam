import os
import cvzone
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition


facemodel = YOLO(r'yolov11n-face.pt')

database_path = r"C:\Users\omgui\Downloads\sciencefair\sciencefair\sample_database"
known_encodings = []
known_names = []

for file in os.listdir(database_path):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(database_path, file)
        img = face_recognition.load_image_file(img_path)
        
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
print(known_names)


# Video capture setup
webcam = 0
picam = "rtsp://ompi3:8554/stream1"
picam1 = "rtsp://192.168.216.254:8554/stream1"
cap = cv2.VideoCapture(picam1)

while True:
    rt, frame = cap.read()
    if not rt:
        break
    
    rgb_frame = frame[:, :, ::-1]  # Convert to RGB
    video_resized = cv2.resize(rgb_frame, (700, 500))
    
    face_result = facemodel.predict(video_resized, conf=0.40)

    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = y2 - y1, x2 - x1

            face_img = video_resized[max(0, y1-50):min(y2+50, video_resized.shape[0]), max(0, x1-50):min(x2+50, video_resized.shape[1])]
            face_img = cv2.resize(face_img, (256, 256))
            

            face_encodings = face_recognition.face_encodings(face_img)

            label = "Unkown"
            if face_encodings:
                detected_encoding = face_encodings[0]
                distances = face_recognition.face_distance(known_encodings, detected_encoding)
                min_distance_index = np.argmin(distances)
                
                if distances[min_distance_index] < 0.6:
                    name = known_names[min_distance_index]
                    label = f"{name} ({distances[min_distance_index]:.2f}) Authorized"
                    
            cvzone.cornerRect(video_resized, [x1, y1, w, h], l=9, rt=3)
            cv2.putText(video_resized, label , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    video_resized = video_resized[:, :, ::-1]
    cv2.imshow('frame', video_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
