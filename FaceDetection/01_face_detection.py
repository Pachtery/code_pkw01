import cv2
import os

face_detector = cv2.CascadeClassifier('Cascade/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture('Video.mp4')
camera.set(3, 640) # set video width
camera.set(4, 480) # set video height

#input id face
face_id = input('Face ID : ')

#Initialize detect face
count = 0
while(True):
    ret,img = camera.read()
    img = cv2.flip(img, 1) # flip video image vertically
    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #scaleFactor = 1.3, minNighbors = 5
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
        cv2.rectangle(img,(x,y),(x+w,y+h),color['blue'],2)
        cv2.putText(img,"Face",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color['green'],1,cv2.LINE_AA)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)
    if(cv2.waitKey(1) & 0xFF== ord('q')):
        break
    elif count >= 1000: 
        break
camera.release()
cv2.destroyAllWindows()
