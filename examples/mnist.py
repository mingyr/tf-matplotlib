# Copyright 2018 Christoph Heindl.
# Copyright 2026 Yurui Ming.
# Licensed under MIT License
# ============================================================
"""Show usage of confusion matrix visualization.

Using a simple MNIST classifier taken from
https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py

Code is modified to slow down convergence so that
time-stepping confusion matrix in Tensorboard has a
better visual effect.
"""

from datetime import datetime
import tensorflow as tf
import numpy as np
import os

import tf_matplotlib as tfmpl

@tfmpl.figure_tensor
def draw_confusion_matrix(matrix):
    '''Draw confusion matrix for MNIST.'''
    fig = tfmpl.create_figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_title('Confusion matrix for MNIST classification')
    
    tfmpl.plots.confusion_matrix.draw(
        ax, matrix,
        axis_labels=['Digit ' + str(x) for x in range(10)],
        normalize=True
    )

    return fig
    
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = tf.cast(y_train, dtype=tf.int32)
    y_test = tf.cast(y_test, dtype=tf.int32)
    
    mnist = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    mnist_iter = iter(mnist.shuffle(1000).batch(32))

    # Create the model
    W = tf.Variable(tf.random.normal([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    optimizer = tf.keras.optimizers.SGD(1e-1)

    os.makedirs('log', exist_ok=True)
    now = datetime.now()
    logdir = "log/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.create_file_writer(logdir)
    
    # Train
    for i in range(1000):            
        batch_xs, batch_ys = next(mnist_iter)
        
        with tf.GradientTape() as tape:
            y = tf.linalg.matmul(tf.reshape(batch_xs, [tf.shape(batch_xs)[0], -1]), W) + b
            loss = tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_ys, logits=y)
            )
        
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply(gradients, [W, b])

        if i % 100 == 0:
            preds = tf.math.argmax(y, axis=-1)

            # Compute confusion matrix
            with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(batch_ys), tf.shape(preds))]):
                matrix = tf.math.confusion_matrix(batch_ys, preds, num_classes=10)

            # Get a image tensor for summary usage
            image_tensor = draw_confusion_matrix(matrix)
            
            with writer.as_default():
                image_summary = tf.summary.image('confusion_matrix', image_tensor, step=i)

    y = tf.linalg.matmul(tf.reshape(x_test, [tf.shape(x_test)[0], -1]), W) + b
    preds = tf.math.argmax(y, 1, output_type=tf.int32)

    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(preds), tf.shape(y_test))]):
        correct_prediction = tf.math.equal(preds, y_test)
    accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy)
    