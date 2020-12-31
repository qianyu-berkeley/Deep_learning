import numpy as np
import Parlour.NN_utils as utils


class NN_model_blocks:
    def __init__(self, config):
        self.layer_dims = config[
            "layer_dims"
        ]  # a list contains the dimension of hidden layers
        self.seed = config[
            "seed"
        ]  # random seeds for NN initialization for repeatable results
        self.lambd = config["lambd"]
        self.keep_prob = config["keep_prob"]
        self._parameters = {}
        self._v = {}
        self._s = {}

    def initialize_parameters_deep_xavier(self):
        """
        Initialize deep NN parameters W, b

        Arguments:
        ----------
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(self.seed)
        L = len(self.layer_dims)

        for l in range(1, L):
            self._parameters["W" + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) / np.sqrt(self.layer_dims[l - 1])
            self._parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert self._parameters["W" + str(l)].shape == (
                self.layer_dims[l],
                self.layer_dims[l - 1],
            )
            assert self._parameters["b" + str(l)].shape == (self.layer_dims[l], 1)

    def initialize_parameters_deep_he(self):
        """
        Initialize deep NN parameters W, b

        Arguments:
        ----------
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(self.seed)
        L = len(self.layer_dims)

        for l in range(1, L):
            self._parameters["W" + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]
            ) * np.sqrt(2.0 / self.layer_dims[l - 1])
            self._parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert self._parameters["W" + str(l)].shape == (
                self.layer_dims[l],
                self.layer_dims[l - 1],
            )
            assert self._parameters["b" + str(l)].shape == (self.layer_dims[l], 1)

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        -----------
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        --------
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ;
                 stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b
        assert Z.shape == (W.shape[0], A.shape[1])

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        ----------
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        --------
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """
        if activation == "sigmoid":  # output layer
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = utils.sigmoid(Z)
            assert A.shape == (W.shape[0], A_prev.shape[1])
            dropout_cache = np.ones(A.shape)
            cache = (linear_cache, activation_cache)
            return A, cache

        elif activation == "relu":  # hidden layer
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = utils.relu(Z)
            assert A.shape == (W.shape[0], A_prev.shape[1])

            # append dropout prob and parameters with dropout to the cache
            AD, dropout_cache = utils.invert_dropout(A, self.keep_prob)
            cache = (linear_cache, activation_cache)
            return AD, cache, dropout_cache

    def L_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Returns:
        --------
        AL -- last post-activation value
        caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        dropout_caches = []
        A = X
        L = len(self._parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache, dropout_cache = self.linear_activation_forward(
                A_prev,
                self._parameters["W" + str(l)],
                self._parameters["b" + str(l)],
                activation="relu",
            )
            caches.append(cache)
            dropout_caches.append(dropout_cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(
            A,
            self._parameters["W" + str(L)],
            self._parameters["b" + str(L)],
            activation="sigmoid",
        )

        caches.append(cache)
        assert AL.shape == (1, X.shape[1])

        return AL, caches, dropout_caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function with minibatch

        Arguments:
        ----------
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        ----------
        cost -- cross-entropy cost
        """
        m = Y.shape[1]

        # Compute cross entropy loss from aL and y.
        # For minibatch, we do not divided by total sample size
        # -sum((Ylog(AL) + (1 - Y)log(1 - AL)))
        # make sure the cost's shape is () (e.g. this turns [[17]] into 17).
        log_probs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        cost_minibatch_total = np.sum(log_probs)  # mini batch

        # L2 Regularization Cost
        # For minibatch, we do not divided by total sample size
        L2_regularization_cost = (
            sum([np.sum(np.square(w)) for w in self._parameters.values()])
            * self.lambd
            / 2
        )

        cost = cost_minibatch_total + L2_regularization_cost
        assert cost.shape == ()

        return cost

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        ----------
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        --------
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1.0 / m * np.dot(dZ, A_prev.T) + (self.lambd / m) * W
        db = 1.0 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, dropout_cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        ----------
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        ---------
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        if activation == "relu":
            linear_cache, activation_cache = cache
            dZ = utils.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            linear_cache, activation_cache = cache
            dZ = utils.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        # Apply mask D to shut down the same neurons as during the
        # back propagation
        D = dropout_cache
        dA_prev = dA_prev * D
        dA_prev = dA_prev / self.keep_prob

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches, dropout_caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        dropout_caches -- list of dropout caches containing:
                    every dropout cache of activation except the final activation where drop out does not apply

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients.
        # Inputs: "dAL, current_cache, None".
        # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        current_dropout_cache = dropout_caches[L - 2]
        (
            grads["dA" + str(L - 1)],
            grads["dW" + str(L)],
            grads["db" + str(L)],
        ) = self.linear_activation_backward(
            dAL, current_cache, current_dropout_cache, "sigmoid"
        )

        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache, dropout_cache".
            # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]

            if l == 0:
                current_dropout_cache = np.ones((1, 1))
            else:
                current_dropout_cache = dropout_caches[l - 1]

            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, current_dropout_cache, "relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...

        Arguments:
        ----------
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward
        """

        L = len(self._parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):

            self._parameters["W" + str(l + 1)] = (
                self._parameters["W" + str(l + 1)]
                - learning_rate * grads["dW" + str(l + 1)]
            )
            self._parameters["b" + str(l + 1)] = (
                self._parameters["b" + str(l + 1)]
                - learning_rate * grads["db" + str(l + 1)]
            )

    def initialize_adam(self):
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl

        Returns:
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """

        L = len(self._parameters) // 2  # number of layers in the neural networks

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            self._v["dW" + str(l + 1)] = np.zeros(
                self._parameters["W" + str(l + 1)].shape
            )
            self._v["db" + str(l + 1)] = np.zeros(
                self._parameters["b" + str(l + 1)].shape
            )
            self._s["dW" + str(l + 1)] = np.zeros(
                self._parameters["W" + str(l + 1)].shape
            )
            self._s["db" + str(l + 1)] = np.zeros(
                self._parameters["b" + str(l + 1)].shape
            )

    def update_parameters_with_adam(
        self, grads, t, learning_rate, beta1, beta2, epsilon
    ):
        """
        Update parameters using Adam

        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates
        beta2 -- Exponential decay hyperparameter for the second moment estimates
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """

        L = len(self._parameters) // 2  # number of layers in the neural networks
        v_corrected = {}  # Initializing first moment estimate, python dictionary
        s_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            self._v["dW" + str(l + 1)] = (
                beta1 * self._v["dW" + str(l + 1)]
                + (1 - beta1) * grads["dW" + str(l + 1)]
            )
            self._v["db" + str(l + 1)] = (
                beta1 * self._v["db" + str(l + 1)]
                + (1 - beta1) * grads["db" + str(l + 1)]
            )

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            v_corrected["dW" + str(l + 1)] = self._v["dW" + str(l + 1)] / (
                1 - beta1 ** t
            )
            v_corrected["db" + str(l + 1)] = self._v["db" + str(l + 1)] / (
                1 - beta1 ** t
            )

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            self._s["dW" + str(l + 1)] = beta2 * self._s["dW" + str(l + 1)] + (
                1 - beta2
            ) * np.square(grads["dW" + str(l + 1)])
            self._s["db" + str(l + 1)] = beta2 * self._s["db" + str(l + 1)] + (
                1 - beta2
            ) * np.square(grads["db" + str(l + 1)])

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            s_corrected["dW" + str(l + 1)] = self._s["dW" + str(l + 1)] / (
                1 - beta2 ** t
            )
            s_corrected["db" + str(l + 1)] = self._s["db" + str(l + 1)] / (
                1 - beta2 ** t
            )

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            self._parameters["W" + str(l + 1)] = self._parameters[
                "W" + str(l + 1)
            ] - learning_rate * v_corrected["dW" + str(l + 1)] / (
                np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon
            )
            self._parameters["b" + str(l + 1)] = self._parameters[
                "b" + str(l + 1)
            ] - learning_rate * v_corrected["db" + str(l + 1)] / (
                np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon
            )
