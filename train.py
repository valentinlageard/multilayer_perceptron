from FeedForwardModel import *
import plotly.graph_objects as go

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
    print(f"Model saved to 'trained_model.pkl'")
  
  # Plot training and validation loss and accuracy
  fig = go.Figure()

  x = np.arange(0, len(model.training_loss_history))

  fig.add_trace(go.Scatter(x=x, y=model.training_loss_history, name='Training loss'))
  fig.add_trace(go.Scatter(x=x, y=model.validation_loss_history, name='Validation loss'))
  fig.show()

  fig = go.Figure()

  x = np.arange(0, len(model.training_accuracy_history))

  fig.add_trace(go.Scatter(x=x, y=model.training_accuracy_history, name='Training accuracy'))
  fig.add_trace(go.Scatter(x=x, y=model.validation_accuracy_history, name='Validation accuracy'))
  fig.show()


if __name__ == "__main__":
  main()