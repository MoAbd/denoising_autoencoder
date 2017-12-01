"""
Elliot Greenlee
2017-11-30
UTK 692 Deep Learning Project 3
"""
import tensorflow as tf
import pickle

import datasets
import stacked_autoencoder

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('corruption_type', 'masking', 'Type of input corruption. ["none", "masking", "salt_and_pepper", "gaussian]')
flags.DEFINE_float('corruption_fraction', 0.1, 'Fraction of the input to corrupt.')
flags.DEFINE_float('weight_stddev', 0.01, 'Weight initialization standard deviation')
flags.DEFINE_float('bias_init', 0.1, 'Bias initialization')
flags.DEFINE_float('learning_rate', 0.000002, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 25, 'Size of each mini-batch.')

assert FLAGS.corruption_type in ['masking', 'salt_and_pepper', 'gaussian', 'none']
assert 0. <= FLAGS.corruption_fraction <= 1.


def main():
    print("Loading Dataset")
    #all_x, training_x, testing_x = datasets.load_cifar10_dataset()
    #validation_x = testing_x[:5000]  # Validation set is the first half of the test set

    all_x = pickle.load(open("pooled_data", 'rb'))
    training_x = all_x[:50000]
    testing_x = all_x[50000:60000]
    validation_x = testing_x[:5000]

    print("Building Model")
    sdae = stacked_autoencoder.StackedDenoisingAutoencoder(corruption_type=FLAGS.corruption_type,
                                                           corruption_fraction=FLAGS.corruption_fraction,
                                                           weight_stddev=FLAGS.weight_stddev,
                                                           bias_init=FLAGS.bias_init,
                                                           learning_rate=FLAGS.learning_rate,
                                                           epochs=FLAGS.epochs,
                                                           batch_size=FLAGS.batch_size)

    print("Training")
    #sdae.train(training_x, validation_x)

    print("Transforming")
    pool = sdae.transform(all_x, "SDAE_model2.meta")
    pickle.dump(pool, open("pooled_data2", "wb"))

    print("Saving Weights")
    weights, biases = sdae.save_weights(all_x, "SDAE_model2.meta")
    pickle.dump(weights, open("weights2", "wb"))
    pickle.dump(biases, open("biases2", "wb"))

if __name__ == '__main__':
    main()
