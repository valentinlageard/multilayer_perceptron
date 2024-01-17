import numpy as np
import pickle

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

class Layer:
  def __init__(self, size, activation_function=None):
    self.size = size
    self.activation_function = None
    if activation_function:
      self.activation_function = activation_function
    self.weights = None
    self.biases = None
  
  def activate(self, activations):
    return self.activation_function(np.dot(self.weights, activations) + self.biases)

  def __repr__(self):
    return f"Layer({self.size}, {self.activation_function}) :\nweights:\n{self.weights}\nbiases:\n{self.biases}\n"

class FeedForwardModel:
  def __init__(self, layers):
    # Initialize layer weights and biases using normal distribution
    for i, layer in enumerate(layers[1:]):
      layer.weights = np.random.randn(layer.size, layers[i].size)
      layer.biases = np.random.randn(layer.size, 1)
    # Store them in the model
    self.layers = layers

  def fit(self, data_train, cost_function, learning_rate=0.01, batch_size=8, epochs=100):
    # for epoch in range(epochs):
    #   # for each example in the batch
    #     # run the model:
    #     # for each layer
    #       # compute a(i+1) = activation(w * a(i) + b)
    #     # evaluate the cost (using binary cross entropy error function)
    #     # compute backpropagation for the example:
    #     # for each layer
    #       # 
    #   # compute the average of adjustments of each parameter
    #   # nudge the model parameters using them 
      pass

  def feedforward(self, input_activations):
    activations = input_activations
    print(f"activations: {activations}")
    for i, layer in enumerate(self.layers[1:]):
      activations = layer.activate(activations)
      print(f"activations: {activations}")
    return activations

  def __repr__(self):
    representation = ""
    for layer in self.layers:
      representation += layer.__repr__()
    return representation

def main():
  model = FeedForwardModel([
    Layer(8),
    Layer(16, sigmoid),
    Layer(2, sigmoid)
  ])
  print(model)

  model.feedforward(np.random.randn(8,1))

if __name__ == '__main__':
  main()
