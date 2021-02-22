# Descartes

# Data
The data is assumed to be in the directory ```auto-insurance-fall-2017```

# Creating docker container
In order, to make sure that the code will work on different machines, I've trained
the model in a docker container.

To create the container, first run ```create_container.sh```. Then, to train the model and make predictions run ```docker_train.sh```

# Description of the directories
```models``` - directory with the trained models
```notebooks``` - directory with jupyter notebooks. There are two of them:
- Exploratory data analysis
- Hyperparameters tuning

```production_tools``` - directory, where the trained label encoder is saved.
```src``` - directory with the source code.

# Metric and model
The metric used for the training is ROC AUC. Later, we could use ROC AUC curve, to choose the most optimal thresholds. In this project, we want the model to be more accurate when it comes to detecting false negatives.

The model used is a simple ensemble of XGboost.

# Validation strategy
A 5-fold stratified validation was used to split the data.
