import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
ptime=0
mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

while True:
    success,frame=cap.read()
    imgRGB= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=pose.process(imgRGB)


    if(res.pose_landmarks):
        mpDraw.draw_landmarks(frame,res.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(res.pose_landmarks.landmark):
            h,w,c=frame.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(frame,(cx,cy),2,(255,0,0),-1)



    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(frame,str(int(fps)),(50,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),4)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)