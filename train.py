from FeedForwardModel import *

def main():
  try:
    with open('training_data.pkl', 'rb') as f:
      training_data = pickle.load(f)
  except IOError as error:
    print(f"Error: training data wasn't generated.")
    return

  try:
    with open('validation_data.pkl', 'rb') as f:
      validation_data = pickle.load(f)
  except IOError as error:
    print(f"Error: validation data wasn't generated.")
    return

  model = FeedForwardModel(
    [Layer(training_data[0].shape[1]),
      Layer(20, SigmoidActivation),
      Layer(20, SigmoidActivation),
      Layer(training_data[1].shape[1], SigmoidActivation)],
    QuadraticLoss)

  model.stochastic_gradient_descent(
    training_data,
    validation_data=validation_data,
    batch_size=16,
    epochs=100,
    learning_rate=0.1,
    patience=5)

  with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

if __name__ == "__main__":
  main()