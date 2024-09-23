from keras.models import Sequential
from keras.layers import Dense
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import Metrics
import numpy as np


def get_model():
    """Constructs a simple model architecture suitable for MNIST."""
    model = Sequential()
    model.add(Dense(12, input_shape=(4,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test) -> None:
        # Create model
        self.model = get_model()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # self.model.fit(self.trainset, epochs=1, verbose=VERBOSE)
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=1)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, np.array(self.y_test), verbose=1)
        print(f"Evaluation loss: {loss}, accuracy: {acc}")
        return loss, len(self.X_test), {"accuracy": acc}

def get_client_fn(X_trains, X_tests, y_trains, y_tests):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        # cid is the dataset id here
        cid = int(cid)
        # Create and return client
        print("CID", cid)
        return FlowerClient(X_trains[cid], y_trains[cid], X_tests[cid], y_tests[cid])

    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(X_test, y_test):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()
        model.set_weights(parameters)
        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        return loss, {"accuracy": acc}

    return evaluate
