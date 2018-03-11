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

images, labels = load_data(train_data_directory)
imagesT, labelsT = load_data(test_data_directory)


def alter_images(imgArray):
    print('kek')
    # Images resized to
    returnImgArray = [transform.resize(image, (150,150)) for image in imgArray]
    # From Python List to Speedy Numpy Array
    returnImgArray = np.array(returnImgArray)
    
    #images28 = rgb2gray(images28)
    
    return returnImgArray
    
images28 = alter_images(images)
imagesTest = alter_images(imagesT)


# Get the unique labels 
unique_labels = set(labels)



print(images28[0][0])

while(True):
    # HARD ASS TENSORFLOW SHIT THAT I HAVE NO FUCKING IDEA ABOUT

    # intialize placeholders
    x = tf.placeholder(dtype = tf.float32, shape = [None, 150, 150, 3], name="X")
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # Intialize placeholders 
    images_flat = tf.contrib.layers.flatten(x)

    # Fully conncted layer
    logits = tf.contrib.layers.fully_connected(images_flat, 3, tf.nn.relu)

    # Define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

    # Define an optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # Convert logits to label indexes
    correct_pred = tf.argmax(logits, 1, name="correct_pred")

    # Define an accuracy metric
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    print("IMAGES_FLAT: ", images_flat)
    print("LOGITS: ", logits)
    print("LOSS: ", loss)
    print("PREDICTED_LABELS: ", correct_pred)


    tf.set_random_seed(1234);
    sess = tf.Session()

    # Let's create a Saver object
    # By default, the Saver handles every Variables related to the default graph
    all_saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())


    for i in range(121):
        print("EPOCH", i);
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        #if(i % 10 == 0):
        print("Loss: ", loss)
        print("reee:" , accuracy_val)
        print("DONE WITH EPOCH")
       
    # CODE TO STOP WHEN ACCURACY IS 0 CONSISTANTLY
    if(accuracy_val>0.3):
        print(x)
        all_saver.save(sess, dir + '/data-all')
        break



    





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

