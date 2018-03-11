import tensorflow as tf
import os
from skimage import data, transform, io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import random

dir = os.path.dirname(os.path.realpath(__file__))


ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"

images = []
#images.append(data.imread("/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/TidePods/Testing/00001/download.jpg"))
images.append(data.imread("/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/vision/sup.jpg"))

editedImgArray = [transform.resize(image, (150,150)) for image in images]
imagesTest = np.array(editedImgArray)


sess = tf.Session()

saver = tf.train.import_meta_graph("/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all.meta")
saver.restore(sess, "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all")
graph = tf.get_default_graph()

x = graph.get_operation_by_name('X').outputs[0]
correct_pred = graph.get_operation_by_name('correct_pred').outputs[0]



# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: imagesTest})[0]

print(predicted) # predicted label number


io.imshow(imagesTest[0])
plt.show()
