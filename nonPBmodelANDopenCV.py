import tensorflow as tf
import cv2
import sys
import os
from skimage import data, transform, io, color
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

sess = tf.Session()

saver = tf.train.import_meta_graph("/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all.meta")
saver.restore(sess, "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all")
graph = tf.get_default_graph()

x = graph.get_operation_by_name('X').outputs[0]
correct_pred = graph.get_operation_by_name('correct_pred').outputs[0]

video_capture = cv2.VideoCapture(0)

while True:
	
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
  
    # OpenCV frames are in BGR and we need it in RGB, so we convert
    b,g,r = cv2.split(frame)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
  
    images = []
  
	# resizing image for Neural Network Model
    gg = transform.resize(rgb_img, (150,150))
    images.append(gg)
	
    imagesTest = np.array(images)

	
    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: imagesTest})[0]
    
    print(predicted) # predicted label number

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


