import cv2
import sys
import os
from skimage import data, transform, io, color
import numpy as np
from PIL import Image


ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"


video_capture = cv2.VideoCapture(0)

while True:
	
    # Capture frame-by-frame
    ret, frame = video_capture.read()
  
    # OpenCV frames are in BGR and we need it in RGB, so we convert
    b,g,r = cv2.split(frame)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
  
	# resizing image for Neural Network Model
    gg = transform.resize(rgb_img, (150,150))
    
    # Testing image data
    io.imsave("sup.jpg",gg)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
