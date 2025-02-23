# Federated Learning with Multiple Datasets

This project implements federated learning using Flower framework with multiple datasets.

## Structure
- `utils.py`: Contains utility functions and FlowerClient implementation

## Requirements
- TensorFlow
- Flower (flwr)
- NumPy
- scikit-learn

## Usage
```bash
python fl.py
```

# Customization

## Model
You can change the model by changing the code of `get_model` function in _utils.py_. You might need to change the calss to `get_weights`, `set_weights`, `fit`, and `evaluate` functions.

See https://flower.ai/docs/ to check the federatred learning configuration process.


