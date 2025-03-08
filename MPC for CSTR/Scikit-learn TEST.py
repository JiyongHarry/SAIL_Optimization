import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.1, size=X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)  # Split the data into 80% training and 20% testing sets

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
    activation="relu",  # Activation function: 'relu' (others: 'tanh', 'logistic')
    solver="adam",  # Optimizer: 'adam' (others: 'sgd', 'lbfgs')
    alpha=0.01,  # L2 regularization term (higher = stronger regularization)
    learning_rate="adaptive",  # Learning rate strategy
    max_iter=1000,  # Number of epochs
    random_state=42,
)

mlp.fit(X_train_scaled, y_train)

y_pred = mlp.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.scatter(X_test, y_test, label="Actual Data", color="blue", alpha=0.6)
plt.scatter(X_test, y_pred, label="Predicted Data", color="red", alpha=0.6)
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("MLP Regressor Predictions")
plt.show()
