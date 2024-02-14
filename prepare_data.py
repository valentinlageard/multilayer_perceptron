import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def main():
  # Load data from csv
  df = pd.read_csv('data.csv', header=None)

  # Add labels to features
  feature_names = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]
  feature_types = ["mean", "std error", "worst"]
  features = [f"{feature_name} {feature_type}" for feature_type in feature_types for feature_name in feature_names]
  df.columns = ["id", "diagnostic"] + features

  # Remove id column and split between inputs and outputs
  inputs = df.iloc[:,2:].to_numpy()
  outputs = df.iloc[:,1:2].to_numpy()

  # Standardize inputs
  scaler = StandardScaler()
  scaler.fit(inputs)
  scaled_inputs = scaler.transform(inputs)

  # Format outputs to one hot vectors
  formatted_outputs = np.zeros((outputs.shape[0], outputs.shape[1] + 1))
  formatted_outputs[outputs.reshape(outputs.shape[0]) == 'M'] = [1, 0]
  formatted_outputs[outputs.reshape(outputs.shape[0]) == 'B'] = [0, 1]

  # Split data into training (2/3) and validation (1/3) sets
  validation_indices = np.random.choice(range(len(inputs)), size=(int(len(inputs) / 3),), replace=False)
  validation_bool_indices = np.zeros(len(inputs), dtype=bool)
  validation_bool_indices[validation_indices] = True
  training_bool_indices = ~validation_bool_indices
  validation_inputs = scaled_inputs[validation_bool_indices]
  validation_outputs = formatted_outputs[validation_bool_indices]
  training_inputs = scaled_inputs[training_bool_indices]
  training_outputs = formatted_outputs[training_bool_indices]
  validation_data = (validation_inputs, validation_outputs)
  training_data = (training_inputs, training_outputs)

  # Save preprocessed data
  with open('training_data.pkl', 'wb') as f:
      pickle.dump(training_data, f)
  with open('validation_data.pkl', 'wb') as f:
      pickle.dump(validation_data, f)

if __name__ == "__main__":
  main()