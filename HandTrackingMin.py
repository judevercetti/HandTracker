import cv2
import time
import mediapipe as mp

# cap = cv2.VideoCapture("http://192.168.42.129:8080/video")
cap = cv2.VideoCapture(0)

pTime = 0

mpHands = mp.solutions.mediapipe.python.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 0:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('my cam', frame)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()