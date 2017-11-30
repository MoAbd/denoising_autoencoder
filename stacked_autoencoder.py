import tensorflow as tf
import numpy as np

import utils


class StackedDenoisingAutoencoder():
    """
    Implementation of a stacked denoising autoencoder using tensorflow
    """

    def __init__(self, corruption_type, corruption_fraction, weight_stddev, bias_init, learning_rate, epochs, batch_size):
        # Input
        self.corruption_type = corruption_type
        self.corruption_fraction = corruption_fraction

        # Initialization
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init

        # Training
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Model Variables
        self.input_data = None
        self.corrupted_input_data = None
        self.weights = None
        self.biases = None
        self.encode = None
        self.pool = None
        self.decode = None
        self.cost = None
        self.train_step = None

        # Build the model
        self.build_model()

    def build_model(self):
        n_features = 3072

        self.input_data = tf.placeholder(tf.float32, shape=[None, n_features])
        self.corrupted_input_data = tf.placeholder(tf.float32, shape=[None, n_features])
        corrupted_image = tf.reshape(self.corrupted_input_data, [-1, 32, 32, 3])

        self.weights = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=self.weight_stddev))
        self.biases = tf.Variable(tf.constant(self.bias_init, shape=[32]))

        # Encode
        with tf.name_scope("Encode"):
            self.encode = tf.nn.relu(
                tf.nn.conv2d(corrupted_image, self.weights, strides=[1, 1, 1, 1], padding='SAME') + self.biases
            )

        # Pool Encoding
        self.pool = tf.nn.max_pool(self.encode, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Decode
        with tf.name_scope("Decode"):
            self.decode = tf.nn.conv2d_transpose(self.encode, self.weights, output_shape=[-1, 32, 32, 3],
                                                 strides=[1, 1, 1, 1])

        # Flatten
        flat_decode = tf.reshape(self.decode, [-1, 3072])

        # Cost
        with tf.name_scope("Cost_Function"):
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - flat_decode)))
            # _ = tf.summary.scalar("mean_squared", self.cost)  # TODO: do i need this?

        # Optimizer
        with tf.name_scope("Optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    """
    def build_model2(self):
        n_features = ???

        self.input_data = tf.placeholder(tf.float32, shape=[None, n_features])
        self.corrupted_input_data = tf.placeholder(tf.float32, shape=[None, n_features])
        corrupted_image = tf.reshape(self.corrupted_input_data, [-1, 32, 32, 3])  # TODO: match previous layer

        self.weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=self.weight_stddev))
        self.biases = tf.Variable(tf.constant(self.bias_init, shape=[64]))

        # Encode
        with tf.name_scope("Encode"):
            self.encode = tf.nn.relu(
                tf.nn.conv2d(corrupted_image, self.weights, strides=[1, 1, 1, 1], padding='SAME') + self.biases
            )

        # Pool Encoding
        self.pool = tf.nn.max_pool(self.encode, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Decode
        with tf.name_scope("Decode"):
            self.decode = tf.nn.conv2d_transpose(self.encode, self.weights, output_shape=[-1, 32, 32, 3],  # TODO: match previous layer
                                                 strides=[1, 1, 1, 1])

        # TODO: Flatten before comparing with input data

        # Cost
        with tf.name_scope("Cost_Function"):
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
            _ = tf.summary.scalar("mean_squared", self.cost)  # TODO: do i need this?

        # Optimizer
        with tf.name_scope("Optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
    """

    def train(self, training_x, validation_x):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            corruption_ratio = np.round(self.corruption_fraction * training_x.shape[1]).astype(np.int)
            for epoch in range(self.epochs):
                print("Epoch: {}".format(epoch))

                x_corrupted = self.corrupt_input(training_x, corruption_ratio)

                shuff = list(zip(training_x, x_corrupted))
                np.random.shuffle(shuff)

                batches = [_ for _ in utils.gen_batches(shuff, self.batch_size)]

                for batch in batches:
                    x_batch, x_corrupted_batch = zip(*batch)
                    sess.run(self.train_step, feed_dict={self.input_data: x_batch,
                                                         self.corrupted_input_data: x_corrupted_batch})

                if epoch % 50 == 0:
                    if validation_x is not None:
                        error = sess.run(self.cost, feed_dict={self.input_data: validation_x,
                                                               self.corrupted_input_data: validation_x})
                        print("Validation cost is {}".format(error))

            # Save the model
            saver = tf.train.Saver()
            saver.save(sess, 'SDAE_model', global_step=self.epochs)

    def corrupt_input(self, data, v):

        if self.corruption_type == 'masking':
            x_corrupted = utils.masking_noise(data, v)

        elif self.corruption_type == 'salt_and_pepper':
            x_corrupted = utils.salt_and_pepper_noise(data, v)

        elif self.corruption_type == 'gaussian':
            x_corrupted = utils.gaussian_noise(data, v)

        elif self.corruption_type == 'none':
            x_corrupted = data

        else:
            x_corrupted = None

        return x_corrupted

    def transform(self):
        # TODO: load saved model
        # TODO: Run against some data until pooling step
        # TODO: return for saving
        return

    def save_weights(self):
        # TODO: load saved model
        # TODO: save the weights and biases to a specific location
        return
