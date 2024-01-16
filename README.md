# Weather Prediction using Neural Networks

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable)


# Introduction
The project aims to predict snowfall in the Lake Michigan area using satellite imagery and meteorological data. It focuses on understanding and modeling the lake effect snow phenomenon, where cold, dry air passes over a warmer body of water, picking up moisture and resulting in significant snowfall. This natural occurrence is particularly prevalent in the Great Lakes region of North America.

<img src="https://github.com/Manuindukuri/Deep-Learning/assets/114769115/611200c9-17ee-41be-9946-04a0eaf50e8f" width="1000" height="800">

# Purpose
The primary goal is to use various parameters from satellite imagery and meteorological data to predict snowfall. This involves analyzing patterns and relationships between environmental factors and snowfall intensity and distribution.

# Data Sources and Preparation
The project utilizes a dataset labeled 'lat_long_1D_labels_for_plotting.csv', which includes geographical coordinates, indicating a spatial aspect to the data analysis. The dataset likely contains satellite imagery data and meteorological readings such as temperature, humidity, wind speed, and direction, which are crucial for studying lake effect snow.

# Methodology
- Data preprocessing
- Handling missing values
- Normalizing data
- Feature extraction
- Reducing dimensionality

# Model Architecture and Training
Given the project's focus on predicting snowfall using time-series data, a Recurrent Neural Network (RNN) or Long Short-Term Memory (LSTM) model is a suitable choice. These models excel at capturing temporal dependencies and patterns in sequential data, which is essential for accurately forecasting weather-related phenomena like snowfall.

# Sampling
SMOTE (Synthetic Minority Over-sampling Technique) algorithm from the ```imblearn``` library to balance the distribution of the *LES_Snowfall* target variable. This is useful for handling imbalanced datasets, where one class (in this case, "LES_Snowfall" values of 1) is significantly underrepresented compared to the other class. The SMOTE algorithm helps to create a more balanced dataset by generating synthetic samples of the minority class.

# A/B Testing
A/B testing in this context involves comparing different model architectures or varying hyperparameters to determine the most effective approach for predicting snowfall. This includes experimenting with different numbers of LSTM layers, varying the number of neurons in each layer, or adjusting learning rates and regularization techniques.

# Results
The results present the outcomes of the model training and A/B testing. This include visualizations like loss curves, accuracy metrics, showing predicted vs. actual snowfall. The results highlight the model's ability to capture the complex dynamics of lake effect snow.
