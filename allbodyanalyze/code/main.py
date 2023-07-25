import cv2
import mediapipe as mp

webcam=cv2.VideoCapture(0)

mp_draw=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while True:
        ret,frame=webcam.read()
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=holistic.process(rgb)

        mp_draw.draw_landmarks(rgb,result.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(rgb, result.face_landmarks)

        mp_draw.draw_landmarks(rgb, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,mp_draw.DrawingSpec((255,0,0),5,1),mp_draw.DrawingSpec((0,0,255),15,20))
        mp_draw.draw_landmarks(rgb, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

        if cv2.waitKey(20) & 0xFF==ord("q"):
            break

        cv2.imshow("pencere",bgr)

webcam.release()
cv2.destroyAllWindows()









