import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description =r'Build a CNN classifier\using MNIST data')
    parser.add_argument('--input-dir',
    dest='input_dir',
    type=str,
    default='./mnist_data',
    help='Directory for storing data'
    )

# create values for weights in each layer
def get_weights(shape):
    data = tf.truncated_normal(shape, stddev=0.1)
    
    return tf.Variable(data)

# function to create biases in each layer
def get_biases(shape):
    data = tf.constant(0.1, shape=shape)

    return tf.Variable(data)

# function creates a layer based on the input shape
def create_layer(shape):

    # get the weights and biases
    W = get_weights(shape)
    b = get_biases(shape[-1])

    return W, b

# function to perform 2D convolution
def convolution_2d(x, W):

    return tf.nn.conv2d(x, W,
    strides=[1,1,1,1],
    padding='SAME'
    )

# 2x2 max pooling operation
def max_pooling(x):
    return tf.nn.max_pool(x,
    ksize=[1,2,2,1],
    strides=[1,2,2,1],
    padding='SAME'
    )

if __name__ == '__main__':
    
    args = build_arg_parser()

    # get the MNIST data
    mnist = input_data.read_data_sets(args.input_dir, one_hot=True)

    '''
    images are 28x28, so create the input layer
    (28*28=784) input layer = 784 neurons
    '''
    x = tf.placeholder(tf.float32, [None, 784])

    '''
    reshape x into a 4D tensor where
    2nd and 3rd dimensions specify image dimensions
    '''
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 1st conv layer extracting 32 features for each 5x5 patch in the image
    W_conv1, b_conv1 = create_layer([5, 5, 1, 32])

    '''
    convolve the image with the weight tensor
    add the bias and then apply the ReLU
    '''
    h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)

    # apply 2x2 max pooling to the output
    h_pool1 = max_pooling(h_conv1)

    # create 2nd conv layer to compute 64 features for 5x5 patch
    W_conv2, b_conv2 = create_layer([5, 3, 32, 64])

    '''
    convolve the image with the weight tensor
    add the bias and then apply the ReLU
    '''
    h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)

    # apply 2x2 max pooling to the output
    h_pool2 = max_pooling(h_conv2)

    '''
    image size is now reduced to 7x7
    create a layer with 1024 neurons
    '''
    W_cf1, b_fc1 = create_layer([7*7*64, 1024])

    # reshape output from previous layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    '''
    multiply thr previous layers output by the
    weight tensor, add bias apply ReLU
    '''
    h_fc1 = tf.nn.relu(tf.matmu1(h_pool2_flat, W_fc1) + b_fc1)

    # create a dropout layer using probability placeholder
    # reduces overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    '''
    define readout(output) layer with 10 output neurons 
    corresponding to 10 classes in our dataset
    '''
    W_fc2, b_fc2 = create_layer([1024, 10])
    y_conv = tf.matmu1(h_fc1_drop, W_fc2) + b_fc2

    # define entropy loss and the optimizer 
    y_loss = tf.placeholder(tf.float32, [None, 10])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_loss))

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # define how entropy should be computed
    predicted = tf.equal(tf.argmax(y_conv, 1),
    tf.argmax(y_loss, 1)
    )
    accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

    # create and run a session
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)

    # start trainining
    num_iterations = 21000
    batch_size = 75

    print("\nTraining model...")

    for i in range(num_iterations):
        # get the next batch of of images
        batch = mnist.train.next_batch(batch_size)

        # print accuracy progress every 50 iterations
        if i % 50 == 0:
            cur_accuracy = accuracy.eva(feed_dict = {
                x: batch[0], y_loss: batch[1], keep_prob: 1.0
            })

            print('Iteration ', i, ', Accuracy =', cur_accuracy)
        
        # train/ run optimizer on the current batch
        optimizer.run(feed_dict = {
            x: batch[0],
            y_loss:batch[1],
            keep_prob: 0.5
        })

        # compute accuracy using test data
        print('Test accuracy =',
        accuracy.eval(feed_dict = {
            x: mnist.test.images,
            y_loss: mnist.test.labels,
            keep_prob: 1.0
        })
        )
