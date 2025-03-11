import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# Record the start time
start_time = time.time()

# Load the dataset
TotalCase = 100
file_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_{TotalCase}_data_CSTR.csv"
data = pd.read_csv(file_path)

# Split the data by case:
unique_cases = data["case"].unique()

# For reproducibility, set the random seed and choose 80% of cases for training
np.random.seed(42)
train_cases = np.random.choice(
    unique_cases, size=int(0.8 * unique_cases.shape[0]), replace=False
)  # should revise
test_cases = np.setdiff1d(unique_cases, train_cases)

# Filter the dataset for training and testing based on selected cases
train_data = data[data["case"].isin(train_cases)]
test_data = data[data["case"].isin(test_cases)]


# Prepare the input and output data
X_train = train_data[["production_target", "time"]]
y_train = train_data["inlet_flow_rate"]
X_test = test_data[["production_target", "time"]]
y_test = test_data["inlet_flow_rate"]

print("Training cases:", train_cases)
print("Testing cases:", test_cases)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Define the MLP regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 neurons each
    activation="relu",  # Activation function: 'relu' (others: 'tanh', 'logistic')
    solver="adam",  # Optimizer: 'adam' (others: 'sgd', 'lbfgs')
    alpha=0.01,  # L2 regularization term
    learning_rate="adaptive",  # Learning rate strategy
    max_iter=1000,  # Maximum number of iterations
    random_state=42,
)

# Train the model on the training cases
mlp.fit(X_train, y_train)

# Make predictions on the test cases
y_pred = mlp.predict(X_test)

# Record the end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2:.4f}")

# Save the trained model
model_path = "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/ann_model.pkl"
joblib.dump(mlp, model_path)
print(f"Model saved to {model_path}")

# Plot the results
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap("jet", len(test_cases))

for i, case in enumerate(test_cases):
    case_data = test_data[test_data["case"] == case]
    case_X_test = case_data[["production_target", "time"]]
    case_y_test = case_data["inlet_flow_rate"]
    case_y_pred = mlp.predict(case_X_test)

    plt.scatter(
        case_data["time"],
        case_y_test,
        label=f"Case {case} y_test",
        color=colors(i),
        alpha=0.6,
    )
    plt.plot(
        case_data["time"], case_y_pred, label=f"Case {case} y_pred", color=colors(i)
    )

plt.xlabel("Time")
plt.ylabel("Inlet Flow Rate")
plt.title("Inlet Flow Rate Over Time for Test Cases")
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
plt.show()
