#Load the MNIST data, which comes pre-loaded with TensorFlow.

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
#
# However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
#
# In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
#

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

## Visualize Data
#View a sample from the dataset.

import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])

## Preprocess Data
# Shuffle the training data.

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

## Setup TensorFlow
#import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

# ## SOLUTION: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
#
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
#
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 14x14x6.
#
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 5x5x16.
#
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
#
# **Layer 3: Fully Connected.** This should have 120 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 4: Fully Connected.** This should have 84 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
#
# ### Output
# Return the result of the 2nd fully connected layer.

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1 #how to initialize weights? experiment with values

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    #Has 5x5 filter with an input depth of 1 and an output depth of 6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    #Initialize bias
    conv1_b = tf.Variable(tf.zeros(6))
    #Convolve filter over the images and add bias at the end
    #Convolution: output_height = (input_height - filter_height  + 1) / vertical stride
    #example (current case): (32 - 5 + 1) / 1 = 28
    #formula works for output_width too = 28
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    # pool output using a 2x2 kernel with a 2x2 stride
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    # layer width equal to the number of classes in our label set
    # these outputs are also known as our logits
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

#Train LeNet to classify MNIST data.
#x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
#we initialize the batch size to None which enables placeholder to accept a batch of any size later on
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
# y stores our labels
y = tf.placeholder(tf.int32, (None))
# one hot ints
one_hot_y = tf.one_hot(y, 10)

## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
# the rate tells Tf how quickly to update the network's weights
rate = 0.001 # experiment with other rates (yet another hyperparameter to tune)

# pass input data to lenet to calculate our logits
logits = LeNet(x)
#softmax_cross_entropy_with_logits: compare those logits to the ground-truth labels and calculate the cross_entropy
#cross_entropy is a measure of how different are the logits to the ground-truth training labels
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#reduce_mean: averages the cross entropy from all the training images
loss_operation = tf.reduce_mean(cross_entropy)
#AdamOptimizer uses the Adam algorithm to minimize the loss function similarly to what Stochastic gradient descent does
#a little more sophisticated than stochastic gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# run minimize function on the optimizer which uses backprop to update the networka and minimize training loss
training_operation = optimizer.minimize(loss_operation)
#this is only a pipeline thus we need to pass data into it in order for it to work

## Model Evaluation Pipeline
# Evaluate how well the loss and accuracy of the model for a given dataset.
# Measures whether a given prediction is correct by comparing the logit prediction
# to the one-hot encoded ground-truth label
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# Calculate the model's overall accuracy by averaging the individual prediction accuracies
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Averages the accuracy of each batch to calculate the total accuracy of the model
def evaluate(X_data, y_data): #take dataset as input
    num_examples = len(X_data) #set init variables
    total_accuracy = 0
    sess = tf.get_default_session() # batches the dataset and runs it through
    for offset in range(0, num_examples, BATCH_SIZE): #the evaluation pipeline
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

## Train the Model
# Run the training data through the training pipeline to train the model.#
# Before each epoch, shuffle the training set.#
# After each epoch, measure the loss and accuracy of the validation set.#
# Save the model after training.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.#
# Be sure to only do this once!#
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
