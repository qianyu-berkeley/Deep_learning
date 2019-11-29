# Deeplearning AI Class from Andrew Ng

## Course 1. Neural Network and Deep Learning

### Key Takeaways
* Neural Network's activation layer is move away from __Sigmoid__ function to __Relu__ to improve computation speed because Relu allows faster Graduient Decent converge
* Faster neural network computation allows deep learning practitioner to iterate quickly thus try more different approaches and new ideas
* __Loss function__ vs __Cost Function__ as a function to find global minimum:
    * __Loss Function__ is in respect to a single sample
    * __Cost Function__ is in respect to whole dataset.
* The intuition of the goal of cost functions are that we want to make prediction as close as the actual.
    e.g. Using logistical regression as example:
    We want to ensure that when true label is 1, the model should predict as large as possible, when true label is 0, the model should predict as small as possible.

* __Computation Graph__ is very useful framework to work out forward and backward propagation. It is analogous to chain rule of derivative in calculus
* Numpy for Neural Network implementation:
   * Avoid rank 1 array, it shall be reshaped and np.sum(keep_dim = True):
    `(5,) => (5, 1)`
   * Leverage assert() function to check matrix shape:
    `assert(a.shape == (5,1)`

* Neural Network is analogous to multiple steps chained (hierchical function) of logistic regression
    - Multiple `z = w .* a_1 + b, a_2 = sigmoid(z))` are connected together

* Vectorize notation:
    - matrix n x m => Vertically (n) is the dimention of layers, horizontally (m) is the sample size

* Sigmoid is an older type of activation function.
    - `Sigmoid` function only make sense to use if the output layer is for binary classifications (output class label of [0, 1])
    - `tanh(z)` is almost always better than `Sigmoid` because it shift Sigmoid curve and centers at 0. This is similar to normalize data which make the learning the next layer easier.
    - Both `Sigmoid` and `tanh` have weakness of having small derivative of Z which cause gradient decent to be converge slowly.
    - Rectify linear unit `Relu` has faster gradient decent with large slop (derivative). `Relu` is the default choice of activation function nowadays. Leaky Relu is also commonly used

* W, dw Z, dz have the same dimension in forward and backward propagation.

* If you initialize the neural network to be 0, all hidden units become symetric, as a result all hidden units just compute the same function over and over which is not very useful

* We initilize W to be random small non-zero number, but inialize b to be 0 is ok. If we initialize w to be large, it will slow down learning since `tanh` or `Sigmoid` will start at very small slope (e.g. For `Sigmoid` slow become very small, when W is large)

* Use matrix dimension to check Neural Network construct is recommended
    - w_l = (n_l, n_l-1) and b_l = (n_l, 1) for layer l
    - dw has the same dimension of w, db same as b
    - Z_l = (n_l, m), A_l = (n_l-1, m) for layer l

* Deep Neural Network Intuition: it using earlier layers to detect simply features first then user later layers to detect more complex features

    e.g. In image recognition
    - layer 1: figure out edges
    - layer 2: finding different part of faces
    - layer 3: recognize or detect face


### General NN Building Methodology

The general methodology to build a Neural Network is to:

    1. Define the neural network structure ( # of input units,  # of hidden units, etc).
    2. Initialize the model's parameters
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)

### Model Tuning
- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data.
- regularization

### Reference

- http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
- https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
- http://scs.ryerson.ca/~aharley/neural-networks/
- http://cs231n.github.io/neural-networks-case-study/


## Course 2: Improving deep Neural Networks

* It is a high iterative process (idea => code => experiment) because it is difficult to guess hyper params the first time.
* Common ML often use train/validate/test split of 60/20/20. In the big data era, dev and test is much smaller percentage of total.

    e.g. 1000000 data, only use the necessary to evaluate e.g. 10000. (98% train, 1% dev, 1% test). This is totally depends on the size of data set

* Make sure dev and test data come from the same distribution (It maybe OK not to have a test set, only dev set)
* Bias/Variance (less of a trade-off in deep learning era)
   - High dimensionality: train set error <> dev set error
   - If training set error  >> human performance (opted (bayes) error), than high bias
   - Comparing train set error with bayes error, then compare train set error with dev set errors.


* General Recipe
   - High Bias based on training data performance? => bigger network and train longer, NN architectures search => until bias lower to be acceptable
   - high variance based on dev set performance => get more data, apply regularization, and NN achitecture search
   - Iterate between first 2 steps with many iterations
   - In deep learning big data era, we don't always need to balance the trade-off because getting bigger network and get more data will improve without hurting the model (as long as we do proper reguliarzation)


* Regularization
   - Although L1 in theory will compress the model size to due to its sparsity, but in practice, L1 is less used and less effective. L2 is used much more often in machine learning
   - NN regularization: Frobenious Norm (sum of square matrix) also called weight decay. It is to set W close to 0 to make a simplier network to prevent overfitting
   - Intuition, take tanh activatioon, if z is small. Since z = Wa + b, then we are mostly in the linear region, make the overall model more linear, thus prevent overfitting
   - Visual examination of regularization: plot J with regularizaiton term. (cost of gradient decent)
   - Drop out regularization:
      - Inverted dropout to ensure the expected value of activation stay the same so it will not impact the test prediction
      - Randomly drop out different hidden unit at different model training iteration
      - No drop out at test time since we are making predict
      - Intuition: do not rely on any single feature, so we spread out the weights by shrink weights similar to L2
      - Computer vision uses drop-out very frequently since usually we don' have a lot of image data
      - With drop-out, the cost curve plot will not be monotonic, it is recommended to turn-off dopout first, plot the cost curve to ensure the general NN implementation works before turn-on drop out

* Other method to prevent overfitting
  - Data augmentation: e.g. image random flip, crop, add fake examples
  - Early stopping: stop when dev set error go back up. (no great to do Orthogonalization)
  - Orthogonalization: only focus on optimization or prevent overfitting

* Normalize Inputs
   - Subtract mean (x = x - u)
   - normalize the variance calculate sigma, then do x / sigma, all feature of different dimension has same scale of variance
   - Apply the same u and sigma to the test set
   - normalize features make cost function (surface) more eventually contour (if not, it will be an elongated contour), make it easier to optimize (converge)

* Vanish/exploding gradient problem: activation decrease or increase exponentially with the depth of the network
  - Solution 1: (partial) careful weight random initialization. Different activation has different techniques. Relu: np.sqrt(2/n^[l-1]), tanh uses Xavier (sqrt(1/n^[l-1])), or sqrt(2/(n^[l-1] * n^[l]))
  - gradient checking:
      - reshape all params (W, b) and its derivative (dW, db) to a big vector theta
      - Only for debug not for training
