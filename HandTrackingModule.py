import cv2
import time
import mediapipe as mp

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame


    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        return lmList

def main():
    pTime, cTime = 0, 0
    # cap = cv2.VideoCapture("http://192.168.42.129:8080/video")
    cap = cv2.VideoCapture(0)
    handDetector = HandDetector()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        frame = handDetector.findHands(frame)
        lmList = handDetector.findPosition(frame)
        print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('my cam', frame)

        if cv2.waitKey(1) == 27:
            break


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
