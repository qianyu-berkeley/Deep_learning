import numpy as np
import deep_NN.NN_utils as utils


class NN_model_blocks(object):

    def __init__(self, config):
        self.layer_dims = config['layer_dims']  # a list contains the dimension of hidden layers
        self.seed = config['seed']  # random seeds for NN initialization for repeatable results
        #self.lambd = 0
        #self.keep_prob = 1
        self._parameters = {}

    def initialize_parameters_deep(self):
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
            self._parameters['W' + str(l)] = np.random.randn(self.layer_dims[l],
                                                             self.layer_dims[l-1])/np.sqrt(self.layer_dims[l-1])
            self._parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert(self._parameters['W' + str(l)].shape ==
                   (self.layer_dims[l], self.layer_dims[l-1]))
            assert(self._parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

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
        assert(Z.shape == (W.shape[0], A.shape[1]))

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
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = utils.sigmoid(Z)

        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = utils.relu(Z)

        # append dropout prob and parameters after dropout to the cache
        #dropout_cache = utils.invert_dropout(A, self.keep_prob)
        # cache.append(dropout_cache)
        # caches.append(cache)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

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
        A = X
        L = len(self._parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev, self._parameters['W' + str(l)],
                self._parameters['b' + str(l)],
                activation="relu")
            caches.append(cache)

            # append dropout prob and parameters after dropout to the cache
            #dropout_cache = utils.invert_dropout(A, self.keep_prob)
            # cache.append(dropout_cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(
            A, self._parameters['W' + str(L)],
            self._parameters['b' + str(L)],
            activation="sigmoid")

        caches.append(cache)
        assert(AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        ----------
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        ----------
        cost -- cross-entropy cost
        """
        m = Y.shape[1]

        # Compute loss from aL and y.
        # make sure the cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cross_entropy_cost = (1./m)*(-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cross_entropy_cost = np.squeeze(cross_entropy_cost)

        # L2 Regularization Cost
        # L2_regularization_cost = sum([np.sum(np.square(w))
        #                              for w in self._parameters.values()])*self.lambd/(2*m)

        cost = cross_entropy_cost  # + L2_regularization_cost
        assert(cost.shape == ())

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

        #dW = 1./m * np.dot(dZ, A_prev.T) + (self.lambd/m)*W
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
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
        linear_cache, activation_cache = cache
        #A, D = dropout_cache

        if activation == "relu":
            dZ = utils.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = utils.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
        # dA_prev = dA_prev * D2
        # dA_prev = dA_prev / self.keep_prob

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

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
        # Inputs: "dAL, current_cache".
        # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
            self.linear_activation_backward(dAL, current_cache, 'sigmoid')

        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 1)], current_cache".
            # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads['dA' + str(l+1)], current_cache, 'relu')
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

        L = len(self._parameters)//2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):

            self._parameters["W"+str(l+1)] = self._parameters["W"+str(l+1)] - \
                learning_rate*grads["dW"+str(l+1)]
            self._parameters["b"+str(l+1)] = self._parameters["b"+str(l+1)] - \
                learning_rate*grads["db"+str(l+1)]
