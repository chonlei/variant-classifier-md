# Variant Classification using MD Simulations

Aim: Applying various classification algorithms to determine deleterious variants from VUS based on features extracted from MD simulations.


### Requirements

The code requires Python (3.6+) and the following dependencies:
[scikit-learn](https://scikit-learn.org/stable/install.html), [tensorflow](https://www.tensorflow.org/install).

To install, navigate to the path where you downloaded this repo and run:
```
$ pip install --upgrade pip
$ pip install .
```

## Structure of the repo

### Main
- `nn.py`: Training neural network classifiers.

### Folders
- `data`: Contains MD simulation data and labels.
- `method`: A module containing useful methods, functions, and helper classes for this project;
            for further details, see [here](./method/README.md).
- `out`: Output of the models.
- `reading`: Contains some relevant papers to read.
