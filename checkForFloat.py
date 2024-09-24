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

num_clients = len(datasets)

def check_datasets_for_floats(datasets):
    for i, dataset_path in enumerate(datasets):
        try:
            data = np.genfromtxt(dataset_path, delimiter=',', skip_header=1, dtype=str)
            
            # Check if all values can be converted to float
            all_floats = True
            for row_index, row in enumerate(data):
                for col_index, value in enumerate(row):
                    try:
                        float(value)
                    except ValueError:
                        all_floats = False
                        print(f"Dataset {i} ({dataset_path}): Non-float value '{value}' found at row {row_index + 2}, column {col_index + 1}.")
                        break
                if not all_floats:
                    break
            
            if all_floats:
                print(f"Dataset {i} ({dataset_path}): All values are floats.")
        
        except Exception as e:
            print(f"Error processing dataset {i} ({dataset_path}): {e}")

check_datasets_for_floats(datasets)