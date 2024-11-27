import cv2
from insightface.app import FaceAnalysis
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


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
model = cnn()
model.load_state_dict(torch.load("cnn2.pth"))

model.eval()


emotion_labels = ['neutral', 'fear', 'happy', 'sad', 'surprise', 'angry']

# Initialize RetinaFace model
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))




cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = app.get(frame)

    # Draw bounding boxes and landmarks
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_tensor = preprocess(face_img).unsqueeze(0)
        with torch.no_grad():
            emotion = model(face_tensor)
        label = emotion_labels[np.argmax(emotion)]
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Draw landmarks (optional)
        #for k, v in face.landmark.items():
        #    cv2.circle(frame, (int(v[0]), int(v[1])), 2, (0, 0, 255), -1)


    cv2.imshow('RetinaFace Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()