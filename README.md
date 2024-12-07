# ECE661 Project: Membership Inference Attacks on Stock Prediction Models

This is a group class project for **ECE661 Computer Engineering Machine Learning and Deep Neural Nets at Duke University**, where we investigate the vulnerabilities of stock prediction models to Membership Inference Attacks (MIA). It's collaborated by Isaac Jacobson, Alex Niculescu, and Guangzhi Su.

The project focuses on training stock price prediction models using **Transformer**, **LSTM**, and **MambaStock** architectures, and evaluating their susceptibility to two types of MIA attacks:
1. **Loss-Based Attack**: Calculates the loss of a target model on potential stock data to infer membership.
2. **Shadow Model Attack**: Trains shadow models on a superset of stock data and uses their outputs to train a membership classification model.

---

## Files Overview

### **`lab.py`**
- Main entry point to run the project.
- Supports both hardcoded values and command-line arguments (see `utils/parse_args` for available options).

### **`data.py`**
- Manages the creation and formatting of datasets for training, testing, and attacks.

### **`mamba.py` & `pscan.py`**
- Define the **MambaStock model**, imported from the [MambaStock repo](https://github.com/zshicode/MambaStock).

### **`models.py`**
- Contains definitions for all the stock prediction model architectures, including:
  - **MambaStock**
  - **LSTM**
  - **Transformer**

### **`attacks.py`**
- Implements two types of Membership Inference Attacks:
  1. **Loss-Based Attack**: Infers membership based on the victim model’s loss on specific stock data.
  2. **Shadow Model Attack**: Trains shadow models on a superset of stock symbols and uses them to build a membership classifier.

### **`utils.py`**
- Includes utility functions for various tasks, such as argument parsing, seed setting, and data handling.

### **`symbols.txt`**
- Contains a list of stock symbols used for the initial training and testing of the models.

### **`graph.txt`**
- A single stock symbol used for graphing predictions.

### **`all_symbols.txt`**
- A superset of stock symbols used to evaluate the success of membership inference attacks.

---

## Required Libraries

This project uses the following libraries. Make sure they are installed in your Python environment:

- **PyTorch**: For building and training machine learning models.
- **NumPy**: For numerical computations.
- **Pandas**: For handling datasets.
- **Matplotlib**: For plotting graphs and visualizations.
- **scikit-learn**: For training classifiers and evaluating metrics.
- **yfinance**: For accessing stock data from Yahoo Finance.
  
  ```bash
  pip install torch
  pip install numpy
  pip install pandas
  pip install matplotlib
  pip install scikit-learn
  pip install yfinance
  
---

## Project Highlights

- **Data Source**: Historical stock data is obtained through the Yahoo Finance API.
- **Architectures**: Models are based on **Transformer**, **LSTM**, and **MambaStock**.
- **Attack Methods**:
  - **Loss-Based Attack**: Simple and direct approach.
  - **Shadow Model Attack**: Advanced method leveraging superset training and membership classifiers.
- **Evaluation**: Compare the performance and vulnerability of different architectures under both attack types.

---

This repository serves as a foundation for exploring privacy vulnerabilities in financial machine learning models and offers a comprehensive toolkit for applying and analyzing Membership Inference Attacks.
