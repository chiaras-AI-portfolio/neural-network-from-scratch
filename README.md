# neural-network-from-scratch

In this project, I wanted to implement a simple neural network from scratch using only NumPy. The goal of this project was to understand each step of the training process explicitly, including forward propagation, loss computation, backpropagation, and gradient descent, rather than relying on a deep-learning framework.

The architecture is a simple feed-forward neural network with 2 inputs, 8 hidden neurons, and the ReLU activation function. I used synthetic data in order to create a simple nonlinear regression problem. The target contains both a quadratic term and an interaction term:

$$
y = 1.5x_1^2 + 0.8x_2 - 0.4x_1x_2 + \epsilon
$$

with Gaussian noise $\epsilon$.

This creates a simple nonlinear problem that the neural network can learn.

# Forward Pass

In forward propagation, the input flows through the hidden layer to compute a prediction.

For each sample, the network computes:

$$
Z_1 = XW_1 + b_1
$$

$$
A_1 = \operatorname{ReLU}(Z_1)
$$

$$
\hat{y} = A_1W_2 + b_2
$$

That means that first the input values and weights are combined using a dot product and then the bias is added. The result is passed through the ReLU activation function and then combined again with another weight matrix and bias to produce the final prediction.

# Backpropagation

During training, the gradients are propagated backwards through the network. These gradients are then used to update the weights and biases with gradient descent.

The goal of gradient descent is to minimize the loss function by adjusting the model parameters step by step.

# Results

The model captures the general nonlinear relationship in the data, with a mean absolute error of approximately 1.18 target units. However, the prediction plot shows that the network tends to overestimate some low target values and underestimate larger target values.

The predictions are compressed toward the center of the target range, suggesting that the current network or training setup does not yet fully capture the extremes of the underlying function.

# Next Steps

Possible extensions:

- numerical gradient checking
- comparison with an equivalent PyTorch model
- mini-batch gradient descent
- additional hidden layers