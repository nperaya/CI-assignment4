import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel("AirQualityUCI.xlsx", na_values=-200)
data.dropna(inplace=True)
print("Data loaded successfully.")

# Prepare the input features and target output
# Input attributes: 3, 6, 8, 10, 11, 12, 13, 14
X = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
# Predicting 5 days ahead
y_5_days_ahead = data.iloc[:, 5].shift(-5).dropna().values
# Predicting 10 days ahead
y_10_days_ahead = data.iloc[:, 5].shift(-10).dropna().values

# Remove last 5 and 10 rows from X to match y lengths
X_5 = X[:-5]
X_10 = X[:-10]

# Normalize the features
X_mean = np.mean(X_5, axis=0)
X_std = np.std(X_5, axis=0)
X_scaled_5 = (X_5 - X_mean) / X_std
X_scaled_10 = (X_10 - X_mean) / X_std

# Define hidden layer sizes (can be modified)
hidden_layer_sizes = [5]  # Example: 1 hidden layer with 5 nodes

class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        prev_size = self.input_size
        for size in self.hidden_layer_sizes:
            weights.append(np.random.rand(prev_size, size))
            prev_size = size
        weights.append(np.random.rand(prev_size, self.output_size))
        return weights

    def forward(self, X):
        self.layers = [X]
        for weight in self.weights:
            X = self.relu(np.dot(X, weight))
            self.layers.append(X)
        return X

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, X):
        return self.forward(X)

# Particle class for PSO
class Particle:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.mlp = MLP(input_size, hidden_layer_sizes, output_size)
        self.position = [w.copy() for w in self.mlp.weights]
        self.velocity = [np.random.rand(*w.shape) * 0.1 for w in self.position]
        self.best_position = self.position
        self.best_error = float("inf")

# PSO Algorithm
def pso(X, y, num_particles, num_iterations):
    particles = [Particle(X.shape[1], hidden_layer_sizes, 1) for _ in range(num_particles)]
    global_best_position = None
    global_best_error = float("inf")

    for _ in range(num_iterations):
        for particle in particles:
            y_pred = particle.mlp.predict(X)
            error = np.mean(np.abs(y - y_pred.flatten()))

            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = [w.copy() for w in particle.position]

            if error < global_best_error:
                global_best_error = error
                global_best_position = [w.copy() for w in particle.position]

            inertia = 0.5
            cognitive = 1.5
            social = 1.5

            for i in range(len(particle.position)):
                r1 = np.random.rand(*particle.position[i].shape)
                r2 = np.random.rand(*particle.position[i].shape)
                particle.velocity[i] = (
                    inertia * particle.velocity[i]
                    + cognitive
                    * r1
                    * (particle.best_position[i] - particle.position[i])
                    + social * r2 * (global_best_position[i] - particle.position[i])
                )
                particle.position[i] += particle.velocity[i]

    return global_best_position

# Cross-validation and MAE calculation
num_folds = 10
fold_size = len(X_scaled_5) // num_folds
mae_results_5_days = []
mae_results_10_days = []

# Cross-validation for 5 days ahead prediction
for fold in range(num_folds):
    test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = list(set(range(len(X_scaled_5))) - set(test_indices))

    X_train = X_scaled_5[train_indices]
    y_train = y_5_days_ahead[train_indices]
    X_test = X_scaled_5[test_indices]
    y_test = y_5_days_ahead[test_indices]

    best_weights = pso(X_train, y_train, num_particles=15, num_iterations=200)

    final_mlp = MLP(X_train.shape[1], hidden_layer_sizes, 1)
    final_mlp.weights = best_weights
    y_test_pred = final_mlp.predict(X_test)

    mae = np.mean(np.abs(y_test - y_test_pred.flatten()))
    mae_results_5_days.append(mae)

    print(f"Fold {fold + 1}/{num_folds} (5 days) - Mean Absolute Error: {mae}")

# Cross-validation for 10 days ahead prediction
for fold in range(num_folds):
    test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = list(set(range(len(X_scaled_10))) - set(test_indices))

    X_train = X_scaled_10[train_indices]
    y_train = y_10_days_ahead[train_indices]
    X_test = X_scaled_10[test_indices]
    y_test = y_10_days_ahead[test_indices]

    best_weights = pso(X_train, y_train, num_particles=15, num_iterations=200)

    final_mlp = MLP(X_train.shape[1], hidden_layer_sizes, 1)
    final_mlp.weights = best_weights
    y_test_pred = final_mlp.predict(X_test)

    mae = np.mean(np.abs(y_test - y_test_pred.flatten()))
    mae_results_10_days.append(mae)

    print(f"Fold {fold + 1}/{num_folds} (10 days) - Mean Absolute Error: {mae}")

# Calculate average MAE
average_mae_5_days = np.mean(mae_results_5_days)
average_mae_10_days = np.mean(mae_results_10_days)
print(f"Average Mean Absolute Error (5 days ahead): {average_mae_5_days}")
print(f"Average Mean Absolute Error (10 days ahead): {average_mae_10_days}")

# Plotting MAE for each fold for 5 days
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_folds + 1), mae_results_5_days, marker="o", linestyle="-", color="b")
plt.title("Mean Absolute Error for Each Fold (5 Days Ahead)")
plt.xlabel("Fold Number")
plt.ylabel("Mean Absolute Error")
plt.xticks(range(1, num_folds + 1))
plt.grid()
plt.savefig("mae_per_fold_5_days.png")
plt.show()

# Plotting MAE for each fold for 10 days
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_folds + 1), mae_results_10_days, marker="o", linestyle="-", color="r")
plt.title("Mean Absolute Error for Each Fold (10 Days Ahead)")
plt.xlabel("Fold Number")
plt.ylabel("Mean Absolute Error")
plt.xticks(range(1, num_folds + 1))
plt.grid()
plt.savefig("mae_per_fold_10_days.png")
plt.show()

# Best weights after final training on the entire dataset for 5 days ahead
best_weights_5_days = pso(X_scaled_5, y_5_days_ahead, num_particles=15, num_iterations=200)
final_mlp_5_days = MLP(X_scaled_5.shape[1], hidden_layer_sizes, 1)
final_mlp_5_days.weights = best_weights_5_days
y_pred_5_days = final_mlp_5_days.predict(X_scaled_5)

# Best weights after final training on the entire dataset for 10 days ahead
best_weights_10_days = pso(X_scaled_10, y_10_days_ahead, num_particles=15, num_iterations=200)
final_mlp_10_days = MLP(X_scaled_10.shape[1], hidden_layer_sizes, 1)
final_mlp_10_days.weights = best_weights_10_days
y_pred_10_days = final_mlp_10_days.predict(X_scaled_10)

# Path to save the graph for actual vs predicted (5 days)
plt.figure(figsize=(12, 6))
plt.plot(y_5_days_ahead, label="Actual Benzene Concentration (5 Days Ahead)", color="blue", alpha=0.5)
plt.plot(y_pred_5_days, label="Predicted Benzene Concentration (5 Days Ahead)", color="yellow", alpha=0.5)
plt.title("Benzene Concentration: Actual vs Predicted (5 Days Ahead)")
plt.xlabel("Samples")
plt.ylabel("Benzene Concentration")
plt.legend()
plt.grid()
plt.savefig("actual_vs_predicted_5_days.png")
plt.show()

# Path to save the graph for actual vs predicted (10 days)
plt.figure(figsize=(12, 6))
plt.plot(y_10_days_ahead, label="Actual Benzene Concentration (10 Days Ahead)", color="blue", alpha=0.5)
plt.plot(y_pred_10_days, label="Predicted Benzene Concentration (10 Days Ahead)", color="orange", alpha=0.5)
plt.title("Benzene Concentration: Actual vs Predicted (10 Days Ahead)")
plt.xlabel("Samples")
plt.ylabel("Benzene Concentration")
plt.legend()
plt.grid()
plt.savefig("actual_vs_predicted_10_days.png")
plt.show()
