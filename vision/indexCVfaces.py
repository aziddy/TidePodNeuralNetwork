import cv2
import sys
import os
from skimage import data, transform, io, color
import numpy as np
from PIL import Image


ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"

cascPath = "/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
  
  
    ff = Image.fromarray(frame)
  

    
    gg = transform.resize(ff, (150,150))
    

    ff.convert('RGB').save("sup.jpg")
    
    # Testing image data
 #   io.imsave("sup.jpg",gg)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
