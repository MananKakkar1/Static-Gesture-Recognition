import cv2 as cv
import mediapipe as mp


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def collectHandData(self, img, writer):
        """Process a frame, optionally write a labeled row.

        Returns (quit, wrote):
          quit -> True if 'q' or 'Q' pressed
          wrote -> True if a row was written this frame
        """
        # Flip for selfie-view and run detection
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.hands.process(imgRGB)

        # Draw and capture landmarks for the first detected hand
        landmarks = None
        if res and res.multi_hand_landmarks:
            hand_lms = res.multi_hand_landmarks[0]
            self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
            landmarks = []
            for p in hand_lms.landmark:
                landmarks.extend([p.x, p.y, p.z])

        # Show the updated frame and read key every frame
        cv.imshow('Video', img)
        key = cv.waitKeyEx(1)
        if key in (ord('q'), ord('Q')):
            return True, False

        wrote = False
        if landmarks is not None and key != -1:
            if key in (ord('o'), ord('O')):
                writer.writerow(landmarks + [0])
                print("Saved Open Hand")
                wrote = True
            elif key in (ord('f'), ord('F')):
                writer.writerow(landmarks + [1])
                print("Saved Fist")
                wrote = True
            elif key in (ord('t'), ord('T')):
                writer.writerow(landmarks + [2])
                print("Saved Thumbs Up")
                wrote = True
            elif key in (ord('p'), ord('P')):
                writer.writerow(landmarks + [3])
                print("Saved Peace")
                wrote = True

        return False, wrote
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)
        return lmList

    def extractFeatures(self, img, dropZ=False, return_img=False):
        img = cv.flip(img, 1)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = self.hands.process(imgRGB)
        if not res or not res.multi_hand_landmarks:
            return (None, img) if return_img else None
        hand_lms = res.multi_hand_landmarks[0]
        landmarks = []
        for p in hand_lms.landmark:
            landmarks.extend([p.x, p.y, p.z])
        if dropZ:
            feats = []
            for i in range(0, 63, 3):
                feats.extend([landmarks[i], landmarks[i + 1]])
        else:
            feats = landmarks
        if return_img:
            self.mpDraw.draw_landmarks(img, hand_lms, self.mpHands.HAND_CONNECTIONS)
            return feats, img
        return feats
