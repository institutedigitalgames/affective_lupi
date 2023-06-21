# Affective LUPI


## Description

AI library for Affect Modelling via the Learning via Privileged Information (LUPI) paradigm. The implementation is based on the "**From the Lab to the Wild: Affect Modeling Via Privileged Information**" [[1]](https://arxiv.org/abs/2305.10919). paper 
## Table of Contents

- [Installation](#installation)
- [Scripts](#scripts)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

To access this repository, please follow the steps below:

1. Clone the repository to your local machine using the following command:
 ```git clone <repository_url>```

2. Obtain the necessary access credentials from the project owner/administrator (if necessary).

## Scripts
| Script Name | Description |
| --- | --- |
| experiment_variables | Sets necessary variables for both Regression and Classification experiments |
| ClassificationModels | Implements the models and dataloaders to be used for the Classification experiment |
| RegressionModels | Implements the models and dataloaders to be used for the Regression experiment |
| Classification | Performs the Classification experiment |
| Regression | Performs the Regression experiment |

**Note:** The above-mentioned script have been implemented using the ***RECOLA DB*** [[2]](https://diuf.unifr.ch/main/diva/recola/). paper  as a reference dataset. Running a custom dataset requires modifications on the ***ClassificationModels*** and ***RegressionModels*** scripts.

## Usage

This section provides guidelines on how to customize the scripts in this repository to run your own experiments. Please follow these instructions carefully:

1. Ensure that you have correctly configured your local Python environment using the following command:
```conda env create -f rquirements.yml```

2. Customize the following functions, classes, and methods to accommodate the modalities of your own datasets:

| Function|  Script Name | Description |
| --- | --- |--- |
| classification_data | ClassificationModels  | Returns the participant ids and a dataframe that contains frame paths and minmax normalized features while discarding ambiguous affect values 
| regression_data | RegressionionModels  | Returns the participant ids and a dataframe that contains frame paths and minmax normalized features along with the raw affect trace values 





| Class|  Script Name | Description |
| --- | --- |--- |
| ClassificationDataset | ClassificationModels  | Transforms the input modalities to torch tensors, constructs the classification labels from the affect values, and handles the data batches.
| RegressionDataset | RegressionModels  | Transforms the input modalities and affect targets to torch tensors and handles the data batches.

| Method|  Class | Script Name| Description |
| --- | --- | ---|--- |
| training_step | Model Classes | ClassificationModels, RegressionModels| Performs a batch step on the training set
| validation_step | Model Classes  | ClassificationModels, RegressionModels| Performs a batch step on the val set
| test_step | Model Classes  | ClassificationModels, RegressionModels| Performs a batch step on the test set




## Contributing

Contributions to this repository are currently not allowed. If you have identified any issues or have concerns regarding the content, don't hesitate to contact the project owner/administrator directly.

