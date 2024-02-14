import numpy as np
import pickle


class SigmoidActivation:
  @staticmethod
  def function(z):
    return 1.0 / (1.0 + np.exp(-z))
  
  @staticmethod
  def derivative(z):
    return SigmoidActivation.function(z) * (1.0 - SigmoidActivation.function(z))


class SoftmaxActivation:
  @staticmethod
  def function(z):
    scaled_z = z - np.max(z) # This is done to avoid e(z) to numerically explode
    return np.exp(scaled_z) / sum(np.exp(scaled_z))
  
  @staticmethod
  def derivative(z):
    identity = np.eye(z.shape[0])
    return SoftmaxActivation.function(z) * (identity - SoftmaxActivation.function(z).T)


class QuadraticLoss:
  @staticmethod
  def function(a, y):
    return 0.5*np.linalg.norm(a-y)**2

  @staticmethod
  def derivative(a, y):
    return a - y


class CrossEntropyLoss:
  @staticmethod
  def function(a, y):
    return -np.sum(y * np.log(a))

  @staticmethod
  def derivative(a, y):
    return -y / a


class Layer:
  def __init__(self, size, activation_function=None):
    self.size = size
    self.activation_function = None
    if activation_function:
      self.activation_function = activation_function
    self.weights = None
    self.biases = None
  

  def activate(self, activations, with_weighted_inputs=False):
    if with_weighted_inputs:
      weighted_inputs = np.dot(self.weights, activations) + self.biases
      return (self.activation_function.function(weighted_inputs), weighted_inputs)
    return self.activation_function.function(np.dot(self.weights, activations) + self.biases)


  def __repr__(self):
    return f"Layer({self.size}, {self.activation_function}) :\nweights:\n{self.weights}\nbiases:\n{self.biases}\n"


class FeedForwardModel:
  def __init__(self, layers, cost_function):
    # Initialize layer weights and biases using normal distribution
    for i, layer in enumerate(layers[1:]):
      layer.weights = np.random.randn(layer.size, layers[i].size)
      layer.biases = np.random.randn(layer.size, 1)
    self.layers = layers
    self.cost_function = cost_function

    self.training_loss_history = []
    self.validation_loss_history = []
    self.training_accuracy_history = []
    self.validation_accuracy_history = []
    
  def feedforward(self, input_activations):
    activations = input_activations
    for i, layer in enumerate(self.layers[1:]):
      activations = layer.activate(activations)
    return activations


  def stochastic_gradient_descent(self, training_data, validation_data=None, learning_rate=0.01, batch_size=8, epochs=100):
    for epoch in range(epochs):
      # Prepare batches
      inputs, outputs = training_data
      n_examples = len(inputs)
      permutation_indices = np.random.permutation(n_examples)
      inputs = inputs[permutation_indices]
      outputs = outputs[permutation_indices]
      batches = [(inputs[i:i+batch_size], outputs[i:i+batch_size]) for i in range(0, n_examples, batch_size)]

      # Apply gradient descent on each batch
      for batch in batches:
        self.gradient_descent(batch, learning_rate)

      # Compute validation metrics and store results
      if validation_data:
        validation_loss, validation_accuracy = self.validate(validation_data)
        self.validation_loss_history.append(validation_loss)
        self.validation_accuracy_history.append(validation_accuracy)

      # Compute training metrics and store results
      training_loss, training_accuracy = self.validate(training_data)
      self.training_loss_history.append(training_loss)
      self.training_accuracy_history.append(training_accuracy)
      if validation_data:
        print(f'Epoch {epoch:03}/{epochs:03} | Training loss: {self.training_loss_history[-1]:.4f}, Validation loss: {self.validation_accuracy_history[-1]:.4f}, Training accuracy: {self.training_accuracy_history[-1]:.4f}, Validation accuracy: {self.validation_accuracy_history[-1]:.4f}')
      else:
        print(f'Epoch {epoch:03}/{epochs:03} | Training loss: {self.training_loss_history[-1]:.4f}, Training accuracy: {self.training_accuracy_history[-1]:.4f}')


  def gradient_descent(self, training_data, learning_rate, validation_data=None, epochs=1):
    for epoch in range(epochs):
      # Initialize accumulators to store estimated gradients
      weigths_gradients_accumulators = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
      biases_gradients_accumulators = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]

      # Compute and accumulate gradients deltas for each example in the training_data
      for example in zip(*training_data):
        # Compute gradients deltas
        weights_gradients_deltas, biases_gradients_deltas = self.backpropagate(example)

        # Accumulate gradients deltas over each sample in training data
        for weights_gradients_accumulator, weights_gradients_delta in zip(weigths_gradients_accumulators, weights_gradients_deltas):
          weights_gradients_accumulator += weights_gradients_delta
        for biases_gradients_accumulator, biases_gradients_delta in zip(weigths_gradients_accumulators, biases_gradients_deltas):
          biases_gradients_accumulator += biases_gradients_delta

      # Update weights and biases based on gradients
      for layer, weights_gradients, biases_gradients in zip(self.layers[1:], weigths_gradients_accumulators, biases_gradients_accumulators):
        scaled_learning_rate = learning_rate / len(training_data)
        layer.weights -= weights_gradients * scaled_learning_rate
        layer.biases -= biases_gradients * scaled_learning_rate
      
      # Compute validation metrics and store results
      if validation_data:
        validation_loss, validation_accuracy = self.validate(validation_data)
        self.validation_loss_history.append(validation_loss)
        self.validation_accuracy_history.append(validation_accuracy)

      # Compute training metrics and store results
      if epochs != 1:
        training_loss, training_accuracy = self.validate(training_data)
        self.training_loss_history.append(training_loss)
        self.training_accuracy_history.append(training_accuracy)
        if validation_data:
          print(f'Epoch {epoch:03}/{epochs:03} | Training loss: {self.training_loss_history[-1]:.4f}, Validation loss: {self.validation_accuracy_history[-1]:.4f}, Training accuracy: {self.training_accuracy_history[-1]:.4f}, Validation accuracy: {self.validation_accuracy_history[-1]:.4f}')
        else:
          print(f'Epoch {epoch:03}/{epochs:03} | Training loss: {self.training_loss_history[-1]:.4f}, Training accuracy: {self.training_accuracy_history[-1]:.4f}')


  def backpropagate(self, example):
    example_input, example_result = example
    example_input = example_input.reshape(-1, 1)
    example_result = example_result.reshape(-1, 1)
    # Initialized gradients caches
    weights_gradients_deltas = [np.zeros(layer.weights.shape) for layer in self.layers[1:]]
    biases_gradients_deltas = [np.zeros(layer.biases.shape) for layer in self.layers[1:]]

    # Forward pass
    activations = example_input
    activations_per_layer = [activations] # activation cache for each layer
    weighted_inputs_per_layer = [] # weighted inputs cache for each non-input layer
    # Feedforward the model but cache activations and weighted inputs per layer
    for layer in self.layers[1:]:
      activations, weighted_inputs = layer.activate(activations, with_weighted_inputs=True)
      activations_per_layer.append(activations)
      weighted_inputs_per_layer.append(weighted_inputs)


    # Backward pass
    # Compute the gradient of the cost function with respect to the output layer
    delta = self.cost_function.derivative(activations_per_layer[-1], example_result)

    # Compute and store the gradient of the cost function with respect to the weighted inputs of the output layer
    output_dadz = self.layers[-1].activation_function.derivative(weighted_inputs_per_layer[-1])
    if output_dadz.shape[1] == 1:
      delta *= output_dadz
    else:
      delta = np.sum(delta * output_dadz, axis=0).reshape(-1,1)
    weights_gradients_deltas[-1] = np.dot(delta, activations_per_layer[-2].T)
    biases_gradients_deltas[-1] = delta

    # Compute and store the gradients of each hidden layers 
    for i, layer in enumerate(reversed(self.layers[1:-1])):
      delta = np.dot(self.layers[-i-1].weights.T, delta)
      delta *= layer.activation_function.derivative(weighted_inputs_per_layer[-i-2])
      weights_gradients_deltas[-i-2] = np.dot(delta, activations_per_layer[-i-3].T)
      biases_gradients_deltas[-i-2] = delta
    return (weights_gradients_deltas, biases_gradients_deltas)


  def validate(self, data):
    # Compute average loss
    losses = [self.cost_function.function(self.feedforward(example_input.reshape(-1, 1)), example_output.reshape(-1,1)) for example_input, example_output in zip(*data)]
    average_loss = sum(losses) / len(losses)

    # Compute accuracy
    successes = 0
    failures = 0
    for example_input, example_output in zip(*data):
      example_input = example_input.reshape(-1,1)
      example_output = example_output.reshape(-1,1)
      prediction = self.feedforward(example_input.reshape(-1, 1))
      if np.unravel_index(np.argmax(prediction, axis=None), prediction.shape) == np.unravel_index(np.argmax(example_output, axis=None), example_output.shape):
        successes += 1
      else:
        failures += 1
    return average_loss, successes / (successes + failures)

  def __repr__(self):
    representation = ""
    for layer in self.layers:
      representation += layer.__repr__()
    return representation
