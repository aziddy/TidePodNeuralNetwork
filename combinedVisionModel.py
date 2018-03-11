import cv2
import sys
import os
from skimage import data, transform, io
from skimage.color import rgb2gray
import numpy as np

ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"

cascPath = "/usr/local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    gg = transform.resize(frame, (400,400))
    gg = np.array(gg)
    
    # Testing image data
    #io.imsave("sup.png",gg)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
