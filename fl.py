import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import flwr as fl
import time
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit,  ShuffleSplit
import matplotlib.pyplot as plt
from utils import get_client_fn, get_evaluate_fn, weighted_average

# put here datasets paths
datasets = [
    'datasets/pima_indians.csv',
    'datasets/pakistan.csv',
    'datasets/iraq.csv',
    'datasets/germany.csv',
    'datasets/china.csv',
]

n_splits = 1
test_size = 0.3
min_available_clients_percentage = 1
min_fit_clients_percentage = 1
fraction_fit=1
fraction_evaluate=1
min_evaluate_clients = 1
num_rounds = 5

X_vals = []
y_vals = []
X_trains = []
X_tests = []
y_trains = []
y_tests = []
i = 0
num_clients = len(datasets)

def check_datasets_for_floats(datasets):
    for i, dataset_path in enumerate(datasets):
        try:
            # Load the dataset, skipping the first row
            data = np.genfromtxt(dataset_path, delimiter=',', skip_header=1, dtype=str)
            
            # Check if all values can be converted to float
            all_floats = True
            for row in data:
                for value in row:
                    try:
                        float(value)
                    except ValueError:
                        all_floats = False
                        break
                if not all_floats:
                    break
            
            if all_floats:
                print(f"Dataset {i} ({dataset_path}): All values are floats.")
            else:
                print(f"Dataset {i} ({dataset_path}): Contains non-float values.")
        
        except Exception as e:
            print(f"Error processing dataset {i} ({dataset_path}): {e}")

check_datasets_for_floats(datasets)

# loading datasets, split into training, testing, and validation
for i in range(num_clients):
    dataset = np.loadtxt(datasets[i], delimiter=',', skiprows=1)
    X = dataset[:,0:4]
    y = dataset[:,4]
    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=530)
    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_trains.append(X_train)
        y_trains.append(y_train)

        stratified_split_validation = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=530)
        for t_index,  v_index in stratified_split_validation.split( X[test_index], y[test_index]):
            X_test, X_validation = X_test[t_index], X_test[v_index]
            y_test, y_validation = y_test[t_index], y_test[v_index]

            X_tests.append(X_test)
            y_tests.append(y_test)
            X_vals.append(X_validation)
            y_vals.append(y_validation)
  
# single centralized validation dataset 
X_validation = np.concatenate(X_vals, axis=0)
y_validation = np.concatenate(y_vals, axis=0)



# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    min_fit_clients=num_clients * min_fit_clients_percentage,
    min_evaluate_clients=min_evaluate_clients,
    min_available_clients=int(
        num_clients * min_available_clients_percentage
    ),
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(X_validation, y_validation),
)
client_resources = {"num_cpus": 1, "num_gpus": 0.0}

# Start simulation
history = fl.simulation.start_simulation(
    client_fn=get_client_fn(X_trains, X_tests, y_trains, y_tests),
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_resources=client_resources,
)


global_accuracy_centralised = history.metrics_centralized["accuracy"]
round = [data[0] for data in global_accuracy_centralised]
acc = [100.0 * data[1] for data in global_accuracy_centralised]
global_accuracy_distributed = history.metrics_distributed["accuracy"]
acc_distributed = [100.0 * data[1] for data in global_accuracy_distributed]
acc_distributed.insert(0, 0)  # Adds 1 at index 0

plt.plot(round, acc, label = 'Centralized')
plt.plot(round, acc_distributed, label='Distributed')
plt.grid()
plt.ylabel("Accuracy (%)")
plt.xlabel("Round")
plt.legend()
plt.yscale("log")
plt.show()
