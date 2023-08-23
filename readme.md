# House Price Prediction with Neural Networks

## Overview
This project involves developing a predictive model for house prices using machine learning techniques, specifically focusing on neural networks. The goal is to enhance prediction accuracy and understand the essential steps of data preprocessing, model building, and evaluation. The project is based on real house sale data from the [Kaggle House Pricing Prediction competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Table of Contents
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Results and Submission](#results-and-submission)
- [Usage](#usage)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Data Processing
The project starts with thorough data preprocessing using Python and Pandas. It involves handling missing values, standardizing numerical features, and transforming categorical variables. The data is split into training and testing sets, and PyTorch tensors are prepared for training.

## Model Architecture
The core of the project is the implementation of a Multi-Layer Perceptron (MLP) model using the PyTorch framework. The architecture includes customizable hyperparameters such as the number of layers, neurons per layer, and dropout rate. Various architectural configurations are explored to optimize the model's performance.

## Model Training
The model is trained using the Adam optimizer and a specialized loss function, RMSLE (Root Mean Squared Logarithmic Error). The training process involves iterating through epochs to minimize the loss and improve prediction accuracy. The model's hyperparameters are tuned through k-fold cross-validation.

## Results and Submission
Upon achieving optimal hyperparameters, the model is utilized to generate predictions for house prices. The predictions are then organized into a submission file in CSV format for participation in the Kaggle competition. This step validates the model's real-world applicability.

## Usage
1. Clone this repository.
2. Install the necessary Python packages using `pip install -r requirements.txt`.
3. Run the project's Jupyter Notebook to access the complete code and analysis.

## Contributors
- [Your Name](https://github.com/pudding134)

## Acknowledgments
Special thanks to the Kaggle community and the instructors of ICT303 for providing valuable resources and guidance throughout this project.
