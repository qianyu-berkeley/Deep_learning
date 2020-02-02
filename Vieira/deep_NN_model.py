import tensorflow as tf
from tensorflow.python.framework import ops
from shared_lib.model_utils import random_mini_batches


class deep_NN_model:

    def __init__(self):
        # model parameters default values
        self.learning_rate = 0.01
        self.num_epochs = 2000
        self.minibatch_size = 32
        self.initialization = 'xavier'
        self.seed = 1
        self.layer_dims = []

        # model parameters
        self._parameter = {}

        # object function cost tracker
        self._costs = []

    def set_hyper_params(self, learning_rate, num_epochs, initialization,
                         layer_dims):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.initialization = initialization
        self.layer_dims = layer_dims,
        self.seed = seed

    def create_placeholders(self, n_x, n_y):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        ----------
        n_x -- scalar, input feature size
        n_y -- scalar, number of classes

        Returns:
        X -- tf placeholder for the data input, of shape [n_x, None]
             and dtype "tf.float32"
        Y -- tf placeholder for the input labels, of shape [n_y, None]
             and dtype "tf.float32"

        Note: None type allow flexibility with different sample size
        """
        X = tf.placeholder(tf.float32, shape=(n_x, None))
        Y = tf.placeholder(tf.float32, shape=(n_y, None))

        return X, Y

    def model(self, X_train, Y_train, X_test, Y_test, print_cost=True):
        """
        Implements a L-layer tensorflow neural network
        (LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
        ----------
        X_train -- training set
        Y_train -- training label
        X_test -- test set
        Y_test -- test label
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs

        Returns:
        --------
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # to be able to rerun the model without overwriting tf variables
        ops.reset_default_graph()
        tf.set_random_seed(1)          # to keep consistent results
        seed = 3                       # to keep consistent results

        # (n_x: input size, m : number of examples in the train set)
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]         # n_y : output size
        costs = []                     # To keep track of the cost

        # Create Placeholders of shape (n_x, n_y)
        X, Y = self.create_placeholders(n_x, n_y)

        # Initialize parameters
        parameters = self.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        ZL = self.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(ZL, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # training loop
            for epoch in range(self.num_epochs):
                epoch_cost = 0.0
                # number of minibatches of size minibatch_size in the train set
                num_minibatches = int(m/self.minibatch_size)
                seed = seed + 1
                minibatches = random_mini_batches(X_train, Y_train, self.minibatch_size, seed)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = self.minibatch

                    # run the graph on a minibatch.
                    _, minibatch_cost = sess.run([optimizer, cost],
                                                 feed_dict={X: minibatch_X,
                                                            Y: minibatch_Y})

                    epoch_cost += minibatch_cost/num_minibatches

                # Print the cost every epoch
                if print_cost is True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost is True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # save the parameters in a variable
            parameters = sess.run(parameters)
            print("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

            return parameters

    def initialize_parameters(self):
        """
        Initializes parameters to build a neural network with tensorflow.

        Returns:
        --------
        """

        L = len(self.layer_dims)

        for l in range(1, L):
            self._parameters['W' + str(l)] = tf.get_variable(name="W"+str(l),
                                                             shape=[25, 12288],
                                                             initializer=tf.contrib.layers.xavier_initializer(seed=self._seed))
            self._parameters['b' + str(l)] = tf.get_variable(name="b"+str(l),
                                                             shape=[25, 1],
                                                             initializer=tf.zeros_initializer())

    def forward_propagation(self, X):
        """
        Implements the forward propagation for the model:
        L x {LINEAR -> RELU} -> LINEAR -> SOFTMAX

        Arguments:
        ----------
        X -- input dataset placeholder, of shape (input_size, sample_size)

        Returns:
        ----------
        Z3 -- the output of the last LINEAR unit
        """

        A = X
        L = len(self._parameters) // 2  # number of layers in the neural network

        # From 1 to L-1 hidden layer
        for l in range(1, L-1):
            A_prev = A
            Z = tf.add(tf.matmul(self._parameters['W' + str(l)],
                                 A_prev), self._parameters['b' + str(l)])
            A = tf.nn.relu(Z)

        ZL = tf.add(tf.matmul(self._parameter['W' + str(L-1)],
                              A), self._parameter['b' + str(L-1)])

        return ZL

    def compute_cost(self, ZL, Y):
        """
        Computes the cost

        Arguments:
        ----------
        ZL -- output of forward propagation (output of the last LINEAR unit)
        Y -- "true" labels vector placeholder, same shape as ZL

        Returns:
        --------
        cost - Tensor of the cost function
        """

        logits = tf.transpose(ZL)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=labels))

        return cost
