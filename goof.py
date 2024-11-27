import cv2
from fer import FER

emotion_detector = FER()

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    result = emotion_detector.detect_emotions(frame)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for face in result:
        emotion, score = max(face['emotions'].items(), key=lambda x: x[1])
        cv2.rectangle(frame, (face['box'][0], face['box'][1]), 
                      (face['box'][0] + face['box'][2], face['box'][1] + face['box'][3]), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion}: {score:.2f}", 
                    (face['box'][0], face['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.imshow("ting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
