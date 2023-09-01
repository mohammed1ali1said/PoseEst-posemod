import cv2
import mediapipe as mp
import time

from mediapipe.tasks.python.components.containers import landmark


class PoseDet():
    def __init__(self, mode=False, ubody=False, smoothness=True, DConf=0.5, TConf=0.5):
        self.mode = mode
        self.ubody = ubody
        self.smoothness = smoothness
        self.DConf = DConf
        self.TConf = TConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,1, self.ubody, self.smoothness, DConf, TConf)

    def findPose(self, frame):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.res = self.pose.process(imgRGB)
        for landmark in self.res.pose_landmarks.landmark:
         self.mpDraw.draw_landmarks(frame, self.res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
         return frame



def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    det = PoseDet()
    while True:
        success, frame = cap.read()
        frame=det.findPose(frame)


        cv2.imshow("Output", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
