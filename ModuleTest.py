import dlib
import cv2 as cv

detector_path = "./detector.svm"
detector = dlib.simple_object_detector(detector_path)

camera = cv.imread("TestOneFace.jpg")

def discern(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dets = detector(gray,1)
    for face in dets:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv.rectangle(img,(left,top),(right,bottom),(0,255,0),2)
        cv.imshow("image",img)

while(1):
    discern(camera)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv.destroyAllWindows()
