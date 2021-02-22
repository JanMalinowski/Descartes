# Descartes

## Data
The data is assumed to be in the directory ```auto-insurance-fall-2017```

## Creating docker container and training the model
In order, to make sure that the code will work on different machines, I've trained
the model in a docker container.

To create the container, first run ```create_container.sh``` (Please note, that you have to be in the main directory to do that). Then, to train the model and make predictions run ```docker_train.sh```. The code also returns the OOF ROC_AUC score of the model.

## Description of the directories
```models``` - directory with the trained models
```notebooks``` - directory with jupyter notebooks. There are two of them:
- Exploratory data analysis
- Hyperparameters tuning

```production_tools``` - directory, where the trained label encoder is saved.
```src``` - directory with the source code.

## Metric and model
The metric used for the training is ROC AUC. Later, we could use ROC AUC curve, to choose the most optimal thresholds. In this project, we want the model to be more accurate when it comes to detecting false negatives.

The model used is a simple ensemble of XGboost.

## Validation strategy
A 5-fold stratified validation was used to split the data.

## What can be improved
- Better hyperparameters tuning (e.g. using Optuna).
- Creating more features.
- Using feature importance for feature selection (NOTE: This works best with L1 regularization).
- Creating another model for predicting target amount and then using it as an additional feature.
