import cvzone
from ultralytics import YOLO
import cv2
import os 
import sys
import torch
import torch.nn as nn
from torchvision import transforms
#sys.stdout = open(os.devnull, 'w')
import numpy as np
from PIL import Image
#from fer import FER
#from deepface import DeepFace
import face_recognition
from databasesetup import known_face_encodings, known_face_names
import time
"""
#---------------------USING FER
emotion_detector = FER()

#---------------------USING CNN WITH DATASET
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size =3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size =3)
        self.fc1 = nn.Linear(12800,512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 6)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#model = cnn()
model = torch.load("MFN_msceleb.pth")
model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


#emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""

video = 0
cap = cv2.VideoCapture(video)

#----------------------------------- YOLO 11 METHOD
facemodel = YOLO(r'sciencefair\yolov11n-face.pt')

database_path = r"C:\Users\omgui\Downloads\sciencefair\sciencefair\sample_database"

#print(known_face_encodings)
print(known_face_names)



while True:
    rt, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.resize(rgb_frame, (256, 256))
    face_locations = face_recognition.face_locations(rgb_frame, model = "HOG")
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"  # Default to "Unknown" if no match is found

        # If there's a match, assign the name of the known person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Label the face with the name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        

    time.sleep(0.1)
    # Display the result frame with recognized faces
    cv2.imshow('Real-time Face Recognition', frame)
    """


    
        
    video = cv2.resize(frame, (700, 500))
    
    face_result = facemodel.predict(video,conf = 0.40)

    for info in face_result:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h,w = y2-y1,x2-x1
            
            face_img = video[y1-50:y2+50, x1-50:x2+50]
            
            #result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            #dominant_emotion = result[0]['dominant_emotion']

            #emotions = emotion_detector.detect_emotions(face_img)
            #if emotions:

                #dominant_emotion, score = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                
            #cv2.putText(video, f"{dominant_emotion}", (x1, y1 - 10), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cvzone.cornerRect(video, [x1, y1, w, h], l=9, rt=3)

                
            #face_tensor = preprocess(face_img).unsqueeze(0)
            #with torch.no_grad():
            #    emotion = model(face_tensor)
            #label = emotion_labels[np.argmax(nn.functional.softmax(emotion, dim = 1))]
            #label = emotion_labels[np.argmax(emotion)]
            #cvzone.cornerRect(video,[x1,y1,w,h],l=9,rt=3)
            #cv2.putText(video, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    
    """
    cv2.imshow('frame', video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

