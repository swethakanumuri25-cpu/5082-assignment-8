# Recurrent Neural Networks

## Sequence Modeling with Hidden State
Recurrent Neural Networks are built for sequential data such as text, audio, or time series. Instead of processing each input independently, an RNN maintains a hidden state that carries information from previous time steps. This makes it possible to model temporal dependence and order. At each step, the network updates its hidden state using the current input and the previous hidden state. That recurrent structure allows the model to remember short-range sequence information.

## Vanishing Gradient Problem
A classic challenge in standard RNNs is the vanishing gradient problem. During backpropagation through time, gradients are repeatedly multiplied across many steps. When these values become very small, earlier parts of the sequence receive little learning signal. As a result, standard RNNs struggle to capture long-range dependencies. This is one reason why basic RNNs often perform poorly when tasks require memory over longer contexts.

## LSTM Improvement over Standard RNN
Long Short-Term Memory networks improve on standard RNNs by introducing a memory cell and gating mechanisms. The forget gate controls what old information to remove, the input gate controls what new information to store, and the output gate controls what information to expose. These gates allow gradients and memory to flow more effectively across long sequences. Because of this design, LSTMs are much better at learning long-term dependencies than standard RNNs.
