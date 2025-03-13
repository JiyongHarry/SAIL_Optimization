import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time
import glob
import os


# # Define the read_data function
# def read_data(file_path):
#     df_p = pd.read_excel(file_path, sheet_name="p_val")  # First tab for p_val_array
#     df_u = pd.read_excel(file_path, sheet_name="u_val")  # Second tab for u_val_array
#     print("Data from p_val sheet:")
#     print(df_p)  # Print all rows
#     print("Data from u_val sheet:")
#     print(df_u)  # Print all rows
#     p_val_array = df_p.values
#     u_val_array = df_u.values
#     return p_val_array, u_val_array


# # Record the start time
# start_time = time.time()

# # Load the dataset
# TotalCase = 10
# file_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/generated_{TotalCase}_data_CSTR.xlsx"
# p_val_array, u_val_array = read_data(file_path)

# # Prepare the input and output data
# X = p_val_array
# y = u_val_array
# print(f"X: {X}")
# print(f"y: {y}")


# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=11  # fixed random_state when it's some integer
# )
# print(f"X_train: {X_train}")
# print(f"X_test: {X_test}")
# print(f"y_train: {y_train}")
# print(f"y_test: {y_test}")

# # Scaler needed?

# mlp = MLPRegressor(
#     hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 and 50 neurons
#     activation="relu",  # Activation function: 'relu' (others: 'tanh', 'logistic')
#     solver="adam",  # Optimizer: 'adam' (others: 'sgd', 'lbfgs')
#     alpha=0.01,  # L2 regularization term (higher = stronger regularization)
#     learning_rate="adaptive",  # Learning rate strategy
#     max_iter=1000,  # Number of epochs
#     random_state=11,  # fixed random_state when it's some integer
# )

# # Train the model
# mlp.fit(X_train, y_train)

# # Record the end time
# end_time = time.time()

# # Save the trained model
# model_path = f"/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/MLPRegressor_model_{TotalCase}_case.pkl"
# joblib.dump(mlp, model_path)

# # Make predictions
# y_pred = mlp.predict(X_test)

# # Calculate and print the elapsed time
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"Root Mean Squared Error: {rmse}")
# print(f"R² Score: {r2:.4f}")

# # Plot
# plt.figure(figsize=(10, 6))
# for i in range(y_test.shape[0]):
#     plt.plot(
#         range(y_test.shape[1]),
#         y_test[i, :],
#         "bo-",
#         label="Actual Data" if i == 0 else "",
#         alpha=0.6,
#     )
#     plt.plot(
#         range(y_test.shape[1]),
#         y_pred[i, :],
#         "r+-",
#         label="Predicted Data" if i == 0 else "",
#         alpha=0.6,
#     )
# plt.legend()
# plt.xlabel("Column Index (j)")
# plt.ylabel("Inlet Flow Rate [m3/s]")
# plt.title("MLP Regressor Predictions (ND)")
# plt.grid(True)

# plot_path = f"/Users/jiyong/Library/CloudStorage/Box-Box/SAIL_Research_JiyongLee/MPC for CSTR/MLPRegressor_y_test_{TotalCase}_case.png"
# plt.savefig(plot_path)
# plt.show()

# Single Case Prediction

X_single = np.array(
    [
        [
            0.2725687,
            0.2672106,
            0.26200005,
            0.25673479,
            0.25134909,
            0.24593344,
            0.24073181,
            0.2361159,
            0.23254383,
            0.23050588,
            0.23046875,
            0.23282248,
            0.23784113,
            0.24565542,
            0.25624484,
            0.26944309,
            0.28495723,
            0.30239499,
            0.32129472,
            0.34115624,
            0.36146742,
            0.38172689,
            0.40146089,
            0.42023638,
            0.43767098,
            0.45344177,
            0.46729279,
            0.47904414,
            0.48859918,
            0.4959518,
            0.50119048,
            0.50449741,
            0.50614178,
            0.50646585,
            0.50586301,
            0.50474888,
            0.50352865,
            0.50256121,
            0.50212693,
            0.50240034,
            0.50343418,
            0.50515598,
            0.50737882,
            0.50982845,
            0.51218033,
            0.51410687,
            0.5153262,
            0.51564825,
            0.51501107,
            0.51350266,
            0.51136374,
        ]
    ]
)

y_ground_truth = np.array(
    [
        [
            2.83981888,
            0.77825887,
            0.6713752,
            0.50525606,
            0.32157692,
            0.17700358,
            0.1293789,
            0.21176621,
            0.4738385,
            0.92910215,
            1.57856229,
            2.41199352,
            3.41145187,
            4.55495721,
            5.82165879,
            7.1941112,
            8.6601839,
            10.21297954,
            11.84854065,
            13.56318345,
            15.34876547,
            17.18856343,
            19.05286631,
            20.89677332,
            22.66020995,
            24.27167383,
            25.65513755,
            26.74140477,
            27.47950609,
            27.84850078,
            27.86468701,
            27.582237,
            27.08756791,
            26.48703436,
            25.89200926,
            25.40337431,
            25.10008204,
            25.03066971,
            25.21053141,
            25.62151281,
            26.21568741,
            26.91991444,
            27.64257677,
            28.28423742,
            28.74815158,
            28.95653699,
            28.86453421,
            28.47359644,
            27.83708574,
            27.05686692,
            26.27227446,
        ]
    ]
)

model_files = glob.glob(
    "/Users/jiyong/Git/SAIL_Optimization/MPC for CSTR/MLPRegressor_model_*.pkl"
)
# Sort model files based on the numerical value in the filename
model_files.sort(key=lambda x: int(os.path.basename(x).split("_")[2]))

# Define five distinguishable reddish colors
colors = ["C1", "C2", "C6", "C8", "#FF0000"]

plt.figure(figsize=(10, 6))
for i, model_file in enumerate(model_files):
    loaded_model = joblib.load(model_file)
    y_single_pred = loaded_model.predict(X_single)
    plt.plot(
        range(y_single_pred.shape[1]),
        y_single_pred.flatten(),
        color=colors[i % len(colors)],  # Cycle through the colors
        linestyle="-",
        label=os.path.basename(model_file),
    )

plt.plot(
    range(y_ground_truth.shape[1]),
    y_ground_truth.flatten(),
    "bo",
    label="Ground Truth",
    alpha=0.6,
)
plt.legend()
plt.xlabel("Time horizon (0 to 10s)")
plt.ylabel("Inlet Flow Rate [m3/s]")
plt.title("MLP Regressor Predictions for Single Case")
plt.grid(True)
plt.show()
