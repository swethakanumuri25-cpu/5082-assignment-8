# Artificial Neural Networks

## Basic Structure
Artificial Neural Networks are machine learning models inspired by connected neurons. A basic ANN consists of an input layer, one or more hidden layers, and an output layer. Each connection carries a weight, and each neuron usually applies an activation function to a weighted sum of inputs. During training, the network learns useful weights so that its outputs align with expected labels or targets. ANNs are strong at learning nonlinear relationships, which makes them useful for classification, regression, and representation learning.

## Forward Pass and Activation Functions
In the forward pass, the model receives an input vector and propagates it through the network layer by layer. At each neuron, a weighted sum is computed and then transformed by an activation function such as ReLU, sigmoid, or tanh. Without nonlinear activations, a deep network would collapse into a linear transformation. ReLU is commonly used in hidden layers because it is simple and helps training in deeper networks. Sigmoid is often used when the output is interpreted as a probability.

## Backpropagation and Learning
Backpropagation is the main learning mechanism for ANNs. After a forward pass produces a prediction, the model compares that prediction with the true output using a loss function. The error is then propagated backward using the chain rule to compute gradients for all trainable weights. An optimizer such as gradient descent or Adam updates the weights in the direction that reduces loss. This repeated process allows the network to gradually improve performance across training iterations.
