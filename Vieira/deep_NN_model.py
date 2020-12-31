import tensorflow as tf
from tensorflow.python.framework import ops
from shared_lib.model_utils import random_mini_batches


class deep_NN_model:
    def __init__(self):
        # model parameters default values
        self.learning_rate = 0.01
        self.num_epochs = 2000
        self.minibatch_size = 32
        self.seed = 1
        self.layer_dims = None

        # model parameters
        self._initializer = tf.contrib.layers.xavier_initializer(seed=self.seed)
        self._parameters = {}
        self._X = None
        self._Y = None
        self._ZL = None

        # object function cost tracker
        self._costs = []

    def set_hyper_params(
        self,
        learning_rate,
        num_epochs,
        initialization,
        minibatch_size,
        layer_dims,
        seed,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.initialization = initialization
        self.minibatch_size = minibatch_size
        self.layer_dims = layer_dims
        self.seed = seed
        if initialization == "xavier":
            self._initializer = tf.contrib.layers.xavier_initializer(seed=self.seed)
        elif initialization == "he":
            self._initializer = tf.contrib.layers.variance_scaling_initializer(
                seed=self.seed
            )

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
        self._X = tf.placeholder(tf.float32, shape=(n_x, None))
        self._Y = tf.placeholder(tf.float32, shape=(n_y, None))

    def initialize_parameters(self):
        """
        Initializes parameters to build a neural network with tensorflow.

        Returns:
        --------
        """

        L = len(self.layer_dims)

        for l in range(1, L):
            self._parameters["W" + str(l)] = tf.get_variable(
                name="W" + str(l),
                shape=[self.layer_dims[l], self.layer_dims[l - 1]],
                initializer=self._initializer,
            )
            self._parameters["b" + str(l)] = tf.get_variable(
                name="b" + str(l),
                shape=[self.layer_dims[l], 1],
                initializer=tf.zeros_initializer(),
            )

    def forward_propagation(self):
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

        A = self._X
        L = len(self._parameters) // 2  # number of layers in the neural network
        # From 1 to L-1 hidden layer
        for l in range(1, L):
            A_prev = A
            Z = tf.add(
                tf.matmul(self._parameters["W" + str(l)], A_prev),
                self._parameters["b" + str(l)],
            )
            A = tf.nn.relu(Z)

        self._ZL = tf.add(
            tf.matmul(self._parameters["W" + str(L)], A), self._parameters["b" + str(L)]
        )

    def compute_cost(self):
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

        logits = tf.transpose(self._ZL)
        labels = tf.transpose(self._Y)
        print("logits", logits.shape)
        print("labels", labels.shape)

        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        )

        return cost

    def model(self, X_train, Y_train, print_cost=True):
        """
        Implements a L-layer tensorflow neural network
        (LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
        ----------
        X_train -- training set
        Y_train -- training label
        X_test -- test set
        Y_test -- test label
        print_cost -- True to print the cost every 100 epochs

        Returns:
        --------
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        # to be able to rerun the model without overwriting tf variables
        ops.reset_default_graph()
        tf.set_random_seed(1)  # to keep consistent results

        # (n_x: input size, m : number of examples in the train set)
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]  # n_y : output size
        costs = []  # To keep track of the cost

        # Create Placeholders of shape (n_x, n_y)
        self.create_placeholders(n_x, n_y)

        # Initialize parameters
        self.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        self.forward_propagation()

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost()

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            cost
        )

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
                num_minibatches = int(m / self.minibatch_size)
                self.seed = self.seed + 1
                minibatches = random_mini_batches(
                    X_train, Y_train, self.minibatch_size, self.seed
                )

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # run the graph on a minibatch.
                    _, minibatch_cost = sess.run(
                        [optimizer, cost],
                        feed_dict={self._X: minibatch_X, self._Y: minibatch_Y},
                    )

                    epoch_cost += minibatch_cost / num_minibatches

                # Print the cost every epoch
                if print_cost is True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost is True and epoch % 5 == 0:
                    costs.append(epoch_cost)

            # save the parameters in a variable
            self._parameters = sess.run(self._parameters)
            print("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(self._ZL), tf.argmax(self._Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self._X: X_train, self._Y: Y_train}))

    def save(self):
        # Return nn model parameter dict object
        return self._parameters

    def predict(self, X, Y):
        """
        This function is used to predict the results of a L-layer neural network.

        Arguments:
        ----------
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        ----------
        p -- predictions for the given dataset X
        """
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(self._ZL), tf.argmax(self._Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({self._X: X, self._Y: Y}))
