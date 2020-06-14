from parinya import LINE
import time
import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/classifier_face.xml')
cascadePath = "Cascade/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
line = LINE('cdAJfHOi8LLFdXY9RtzEnLTO33cOid62cpvKr69PNDc')

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'Precha', 'Steve Job','Toey'] 

# Initialize and start realtime video capture
camera = cv2.VideoCapture(0)
camera.set(3, 640) # set video widht
camera.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*camera.get(3)
minH = 0.1*camera.get(4)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID') # กำหนด FourCC code
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (640,480))


while (True):
    ret, img =camera.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # write the flipped frame
    out.write(img)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
        cv2.rectangle(img,(x,y),(x+w,y+h),color['green'], 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 80):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(confidence)
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print("unknown : ",confidence)
            line.sendtext("ผู้คนแปลกหน้า")
            line.sendimage(img[:, :, ::-1])
            time.sleep(5)
            
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, color['white'], 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    if(cv2.waitKey(1) & 0xFF== ord('q')):
            break

camera.release()
out.release()
cv2.destroyAllWindows()
