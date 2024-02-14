import pickle
from FeedForwardModel import *


def main():
  try:
    with open('validation_data.pkl', 'rb') as f:
      validation_data = pickle.load(f)
  except IOError as error:
    print(f"Error: validation data wasn't generated.")
    return

  try:
    with open('trained_model.pkl', 'rb') as f:
      model = pickle.load(f)
  except IOError as error:
    print(f"Error: model wasn't trained.")
    return

  # Compute accuracy
  successes = 0
  failures = 0
  for i, (example_input, example_output) in enumerate(zip(*validation_data)):
    example_input = example_input.reshape(-1,1)
    example_output = example_output.reshape(-1,1)
    prediction = model.feedforward(example_input.reshape(-1, 1))
    loss = CrossEntropyLoss.function(prediction, example_output)
    true_diagnostic = "CANCER" if example_output[0,0] == 1 else "BENIGN"
    predicted_diagnostic = "CANCER" if prediction[0,0] > prediction[1,0] else "BENIGN"
    success_str = ""
    if np.unravel_index(np.argmax(prediction, axis=None), prediction.shape) == np.unravel_index(np.argmax(example_output, axis=None), example_output.shape):
      successes += 1
      success_str = "SUCCESS"
    else:
      failures += 1
      success_str = "FAILURE"
    print(f'Validation example {i:03} | Truth: {true_diagnostic}, Predicted: {predicted_diagnostic} | {"SUCCESS" if true_diagnostic == predicted_diagnostic else "FAILURE"} | Cross entropy loss: {loss:.4f}')

  print(f"Validation accuracy: {(successes / (successes + failures)) * 100:.2f}%")


if __name__ == "__main__":
  main()