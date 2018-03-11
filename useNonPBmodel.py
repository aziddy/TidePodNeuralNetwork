import tensorflow as tf
import os
from skimage import data, transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import random

dir = os.path.dirname(os.path.realpath(__file__))

# loading image data from directories into arrays
def load_data(data_directory):
    directories = []
    for d in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, d)):
            directories.append(d)
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = []
        for f in os.listdir(label_directory):
            if f.endswith(".jpg"):
                file_names.append(os.path.join(label_directory, f))

        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/"
train_data_directory = os.path.join(ROOT_PATH, "TidePods/Training")
test_data_directory = os.path.join(ROOT_PATH, "TidePods/Testing")

imagesT, labelsT = load_data(test_data_directory)


def alter_images(imgArray):
    print('kek')
    # Images resized to
    returnImgArray = [transform.resize(image, (50,50)) for image in imgArray]
    # From Python List to Speedy Numpy Array
    returnImgArray = np.array(returnImgArray)
    
    return returnImgArray
    
imagesTest = alter_images(imagesT)

sess = tf.Session()

saver = tf.train.import_meta_graph("/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all.meta")
saver.restore(sess, "/Users/alex/Desktop/TenserFlowPythonProject/TidePodNeuralNetwork/data-all")
graph = tf.get_default_graph()       
x = graph.get_operation_by_name('X').outputs[0]
correct_pred = graph.get_operation_by_name('correct_pred').outputs[0]


# Pick 10 random images
sample_indexes = random.sample(range(len(imagesTest)), 10)
# Store the 10 images using the randomed indexs
sample_images = [imagesTest[i] for i in sample_indexes]
# Store the 10 asscotiated labels using the randomed indexs
sample_labels = [labelsT[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                       
# Print the real and predicted labels
print(sample_labels)
print(predicted) # predicted label number


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()
