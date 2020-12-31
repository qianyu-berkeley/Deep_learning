import numpy as np
from Parlour.NN_model_blocks import NN_model_blocks
from shared_lib.model_utils import random_mini_batches


class deep_NN_model:
    def __init__(self):

        # model parameters default values
        self.learning_rate = 0.01
        self.num_epochs = 2500
        self.initialization = "xavier"
        self.optimizer = "gd"
        self.minibatch_size = 32
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self._config = {}
        self.seed = 3

        # object function cost tracker
        self._costs = []

        # NN model block definition
        self._nn_model_blocks = None

    def set_hyper_params(
        self,
        num_epochs,
        optimizer_params,
        learning_rate,
        minibatch_size,
        initialization,
        lambd,
        keep_prob,
        layer_dims,
        seed,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.initialization = initialization
        self.seed = seed
        self.optimizer = optimizer_params["optimizer"]
        if self.optimizer == "adam":
            self.beta1 = optimizer_params["beta1"]
            self.beta2 = optimizer_params["beta2"]
            self.epsilon = optimizer_params["epsilon"]
        self._config = {
            "layer_dims": layer_dims,
            "seed": self.seed,
            "lambd": lambd,
            "keep_prob": keep_prob,
        }

    def L_layer_model(self, train_X, train_Y, print_cost=False):
        """
        Construct a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        and train the model with given hyper-parameters

        Returns:
        --------
        """

        # Define NN model blocks
        self._nn_model_blocks = NN_model_blocks(self._config)

        # Total number of data samples
        m = train_X.shape[1]

        # Parameters initialization
        if self.initialization == "xavier":
            self._nn_model_blocks.initialize_parameters_deep_xavier()
        elif self.initialization == "he":
            self._nn_model_blocks.initialize_parameters_deep_he()

        # Loop (gradient descent)
        for i in range(0, self.num_epochs):

            # random minibatches: increment the seed to reshuffle differently
            # the dataset after each epoch
            # self.seed = self.seed + 1
            minibatches = random_mini_batches(
                train_X, train_Y, self.minibatch_size, self.seed
            )
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches, dropout_caches = self._nn_model_blocks.L_model_forward(
                    minibatch_X
                )

                # Compute cost
                cost_total += self._nn_model_blocks.compute_cost(AL, minibatch_Y)

                # Backward propagation
                grads = self._nn_model_blocks.L_model_backward(
                    AL, minibatch_Y, caches, dropout_caches
                )

                # Update parameters.
                self._nn_model_blocks.update_parameters(grads, self.learning_rate)

            # Print the avg cost every 100 training example
            cost_avg = cost_total / m
            if print_cost and i % 100 == 0:
                print("Cost after iteration {:d}: {:f}".format(i, cost_avg))
            if print_cost and i % 100 == 0:
                self._costs.append(cost_avg)

    def adam_model(self, train_X, train_Y, print_cost=False):
        """
        Construct a neural network with Adam optimizer: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
        and train the model with given hyper-parameters

        Returns:
        --------
        """
        # Define NN model blocks
        self._nn_model_blocks = NN_model_blocks(self._config)

        m = train_X.shape[1]  # Total number of data samples
        t = 0  # counter for adam update

        # Parameters initialization
        if self.initialization == "xavier":
            self._nn_model_blocks.initialize_parameters_deep_xavier()
        elif self.initialization == "he":
            self._nn_model_blocks.initialize_parameters_deep_he()

        self._nn_model_blocks.initialize_adam()

        # Optimization loop
        for i in range(self.num_epochs):

            # random minibatches: increment the seed to reshuffle differently
            # the dataset after each epoch
            self.seed = self.seed + 1
            minibatches = random_mini_batches(
                train_X, train_Y, self.minibatch_size, self.seed
            )
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                AL, caches, dropout_caches = self._nn_model_blocks.L_model_forward(
                    minibatch_X
                )

                # Compute cost
                cost_total += self._nn_model_blocks.compute_cost(AL, minibatch_Y)

                # Backward propagation
                grads = self._nn_model_blocks.L_model_backward(
                    AL, minibatch_Y, caches, dropout_caches
                )

                # Update parameters
                t = t + 1  # Adam counter
                self._nn_model_blocks.update_parameters_with_adam(
                    grads, t, self.learning_rate, self.beta1, self.beta2, self.epsilon
                )
            cost_avg = cost_total / m

            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost_avg))
            if print_cost and i % 1000 == 0:
                self._costs.append(cost_avg)

    def training_report(self):
        pass

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
        m = X.shape[1]
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches, dropout_caches = self._nn_model_blocks.L_model_forward(X)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        # print results
        print("Accuracy: {}".format(str(np.sum((p == Y) / m))))

        return p

    def save(self):
        # Return nn model parameter dict object
        return self._nn_model_blocks._parameters
