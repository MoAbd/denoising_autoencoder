import tensorflow as tf
import numpy as np
import os

import utils


class DenoisingAutoencoder(object):

    """ Implementation of Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, model_name='dae', n_components=256, main_dir='dae/', enc_act_func='tanh',
                 dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10,
                 xavier_init=1, opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none',
                 corr_frac=0., verbose=1, seed=-1):
        """
        :param main_dir: main directory to put the models, data and summary directories
        :param n_components: number of hidden units
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'none']
        :param loss_func: Loss function. ['mean_squared', 'cross_entropy']
        :param xavier_init: Value of the constant for xavier weights initialization
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param corr_type: Type of input corruption. ["none", "masking", "salt_and_pepper", "gaussian"]
        :param corr_frac: Fraction of the input to corrupt.
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        self.model_name = model_name
        self.n_components = n_components
        self.main_dir = main_dir
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.xavier_init = xavier_init
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.verbose = verbose
        self.seed = seed

        if self.seed >= 0:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        self.models_dir, self.data_dir, self.tf_summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.input_data = None
        self.input_data_corr = None

        self.W_ = None
        self.bh_ = None
        self.bv_ = None

        self.encode = None
        self.decode = None

        self.train_step = None
        self.cost = None

        self.tf_session = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_saver = None

        self.old_err = 2000

        self.build_mode()

    """Build Model"""
    def build_mode(self):
        """ Creates the computational graph.

        :return: self
        """
        n_features = 3072

        self.input_data = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_data_corr = tf.placeholder('float', [None, n_features], name='x-corr-input')

        self.W_ = tf.Variable(utils.xavier_init(n_features, self.n_components, self.xavier_init), name='enc-w')
        self.bh_ = tf.Variable(tf.zeros([self.n_components]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.zeros([n_features]), name='visible-bias')

        # Encode
        with tf.name_scope("W_x_bh"):
            if self.enc_act_func == 'sigmoid':
                self.encode = tf.nn.sigmoid(tf.matmul(self.input_data_corr, self.W_) + self.bh_)

            elif self.enc_act_func == 'tanh':
                self.encode = tf.nn.tanh(tf.matmul(self.input_data_corr, self.W_) + self.bh_)

            elif self.enc_act_func == 'relu':
                self.encode = tf.nn.relu(tf.matmul(self.input_data_corr, self.W_) + self.bh_)
            else:
                self.encode = None

        # Decode
        with tf.name_scope("Wg_y_bv"):
            if self.dec_act_func == 'sigmoid':
                self.decode = tf.nn.sigmoid(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)

            elif self.dec_act_func == 'tanh':
                self.decode = tf.nn.tanh(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)

            elif self.dec_act_func == 'relu':
                self.decode = tf.nn.relu(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)

            elif self.dec_act_func == 'none':
                self.decode = tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_

            else:
                self.decode = None

        # Cost Function
        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = - tf.reduce_sum(self.input_data * tf.log(self.decode))
                _ = tf.summary.scalar("cross_entropy", self.cost)

            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
                _ = tf.summary.scalar("mean_squared", self.cost)

            else:
                self.cost = None

        # Train Step
        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            elif self.opt == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            else:
                self.train_step = None

    """Train Model"""
    def fit(self, train_set, validation_set=None, restore_previous_model=False):
        """ Fit the model to the data.

        :param train_set: Training data.
        :param validation_set: optional, default None. Validation data.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        self.tf_merged_summaries = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        self.tf_summary_writer = tf.summary.FileWriter(self.tf_summary_dir, self.tf_session.graph)

    def _train_model(self, train_set, validation_set):

        """Train the model.

        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        corruption_ratio = np.round(self.corr_frac * train_set.shape[1]).astype(np.int)

        for i in range(self.num_epochs):
            print("Epoch: {}".format(i))

            self._run_train_step(train_set, corruption_ratio)

            if i % 100 == 0:
                self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)
                if validation_set is not None:
                    err = self._run_validation_error_and_summaries(i, validation_set)

                    if (self.old_err - err) < 0.00005:
                        break

                    self.old_err = err

        self._run_validation_error_and_summaries(i, validation_set)

    def _run_train_step(self, train_set, corruption_ratio):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.

        :param train_set: training set
        :param corruption_ratio: fraction of elements to corrupt

        :return: self
        """
        x_corrupted = self._corrupt_input(train_set, corruption_ratio)

        shuff = list(zip(train_set, x_corrupted))
        np.random.shuffle(shuff)

        batches = [_ for _ in utils.gen_batches(shuff, self.batch_size)]

        for batch in batches:
            x_batch, x_corr_batch = zip(*batch)
            tr_feed = {self.input_data: x_batch, self.input_data_corr: x_corr_batch}
            self.tf_session.run(self.train_step, feed_dict=tr_feed)

    def _corrupt_input(self, data, v):

        """ Corrupt a fraction 'v' of 'data' according to the
        noise method of this autoencoder.
        :return: corrupted data
        """

        if self.corr_type == 'masking':
            x_corrupted = utils.masking_noise(data, v)

        elif self.corr_type == 'salt_and_pepper':
            x_corrupted = utils.salt_and_pepper_noise(data, v)

        elif self.corr_type == 'gaussian':
            x_corrupted = utils.gaussian_noise(data, v)

        elif self.corr_type == 'none':
            x_corrupted = data

        else:
            x_corrupted = None

        return x_corrupted

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """

        vl_feed = {self.input_data: validation_set, self.input_data_corr: validation_set}
        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=vl_feed)
        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, err))

        return err

    def transform(self, data, name='train', save=False):
        """ Transform data according to the model.

        :param data: Data to transform
        :param name: Identifier for the data that is being encoded
        :param save: If true, save data to disk

        :return: transformed data
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            encoded_data = self.encode.eval({self.input_data_corr: data})

            decoded_data = self.decode.eval({self.input_data_corr: data})

            if save:
                np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

            return encoded_data, decoded_data

    def load_model(self, shape, model_path):
        """ Restore a previously trained model from disk.

        :param shape: tuple(n_features, n_components)
        :param model_path: path to the trained model

        :return: self, the trained model
        """
        self.n_components = shape[1]

        self.build_mode(shape[0])

        init_op = tf.global_variables_initializer()

        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)

            self.tf_saver.restore(self.tf_session, model_path)

    def get_model_parameters(self):
        """ Return the model parameters in the form of numpy arrays.

        :return: model parameters
        """
        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            return {
                'enc_w': self.W_.eval(),
                'enc_b': self.bh_.eval(),
                'dec_b': self.bv_.eval()
            }

    def _create_data_directories(self):

        """ Create the three directories for storing respectively the models,
        the data generated by training and the TensorFlow's summaries.

        :return: tuple of strings(models_dir, data_dir, logs_dir)
        """

        self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

        models_dir = 'models/' + self.main_dir
        data_dir = 'data/' + self.main_dir
        logs_dir = 'logs/' + self.main_dir

        for d in [models_dir, data_dir, logs_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        return models_dir, data_dir, logs_dir

    def get_weights_as_images(self, width, height, outdir='img/', max_images=10, model_path=None):
        """ Save the weights of this autoencoder as images, one image per hidden unit.
        Useful to visualize what the autoencoder has learned.

        :type width: int
        :param width: Width of the images

        :type height: int
        :param height: Height of the images

        :type outdir: string, default 'data/sdae/img'
        :param outdir: Output directory for the images. This path is appended to self.data_dir

        :type max_images: int, default 10
        :param max_images: Number of images to return.
        """
        assert max_images <= self.n_components

        outdir = self.data_dir + outdir

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with tf.Session() as self.tf_session:

            if model_path is not None:
                self.tf_saver.restore(self.tf_session, model_path)
            else:
                self.tf_saver.restore(self.tf_session, self.models_dir + self.model_name)

            enc_weights = self.W_.eval()

            perm = np.random.permutation(self.n_components)[:max_images]

            for p in perm:

                enc_w = np.array([i[p] for i in enc_weights])
                image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
                utils.gen_image(enc_w, width, height, image_path)
