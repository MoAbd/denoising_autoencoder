import tensorflow as tf

import autoencoder
import datasets

import matplotlib.pyplot as plt
import numpy as np
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('model_name', 'dae', 'Model name.')
flags.DEFINE_integer('seed', 1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')

# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
flags.DEFINE_string('corr_type', 'masking', 'Type of input corruption. ["none", "masking", "salt_and_pepper", "gaussian]')
flags.DEFINE_float('corr_frac', 0.1, 'Fraction of the input to corrupt.')
flags.DEFINE_integer('xavier_init', 1, 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh", "relu"]')
flags.DEFINE_string('dec_act_func', 'tanh', 'Activation function for the decoder. ["sigmoid", "tanh", "relu", "none"]')
flags.DEFINE_string('main_dir', 'dae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'mean_squared', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('verbose', 1, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('weight_images', 0, 'Number of weight images to generate.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 3000, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 25, 'Size of each mini-batch.')

assert FLAGS.enc_act_func in ['sigmoid', 'tanh', 'relu']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'relu', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'gaussian', 'none']
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']

if __name__ == '__main__':

    all_x, training_x, testing_x = datasets.load_cifar10_dataset()
    validation_x = testing_x[:5000]  # Validation set is the first half of the test set

    # Create the object
    dae = autoencoder.DenoisingAutoencoder(
        seed=FLAGS.seed, model_name=FLAGS.model_name, n_components=FLAGS.n_components,
        enc_act_func=FLAGS.enc_act_func, dec_act_func=FLAGS.dec_act_func, xavier_init=FLAGS.xavier_init,
        corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        loss_func=FLAGS.loss_func, main_dir=FLAGS.main_dir, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
        verbose=FLAGS.verbose, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size)

    # Fit the model
    dae.fit(training_x, testing_x, restore_previous_model=FLAGS.restore_previous_model)

    # Encode the training data and store it
    #dae.transform(training_x, name='train', save=FLAGS.encode_train)
    #dae.transform(validation_x, name='validation', save=FLAGS.encode_valid)
    encoded, decoded = dae.transform(all_x, name='all', save=True)
    pickle.dump(decoded, open("denoised_CIFAR10", "wb"))

    _, denoised = dae.transform(testing_x, name='test', save=FLAGS.encode_test)

    single_img = denoised[0]
    single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
    plt.imshow(single_img_reshaped)
    plt.savefig('full.png')

    # save images
    #dae.get_weights_as_images(28, 28, max_images=FLAGS.weight_images)


