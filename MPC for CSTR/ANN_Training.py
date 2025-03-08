import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pickle

# Load the dataset
file_path = "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_data_CSTR.csv"
data = pd.read_csv(file_path)

# Prepare the input and output data
X = data[["production_target"]]
y = data["inlet_flow_rate"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the trained model
model_path = "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/ann_model.pkl"
joblib.dump(mlp, model_path)
print(f"Model saved to {model_path}")
