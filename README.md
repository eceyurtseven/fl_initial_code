# fl-muliple-datasets
This repository contains code to run a Federated Learning (FL) experiment with the Flower library. The code is structured to easily support multiple datasets, and can be adapted to work with any machine learning model. Each client in the FL setup corresponds to a dataset. Each dataset is splitt into 70% for training, 20% for testing, and 10% for validation. Validation datasets are combined together, in order to be used in validating the aggregated model.

# Installation
```bash
pip install -r requirements.txt
```

# Usage
```bash
python fl.py
```

# Customization

## Model
You can change the model by changing the code of `get_model` function in _utils.py_. You might need to change the calss to `get_weights`, `set_weights`, `fit`, and `evaluate` functions.

See https://flower.ai/docs/ to check the federatred learning configuration process.


