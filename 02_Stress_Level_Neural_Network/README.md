# Neural Network From Scratch: Stress Level Classification

## Overview
This project implements a Multi-Layer Perceptron (MLP) Artificial Neural Network entirely from scratch using only Python and NumPy. The objective was to classify the perceived stress levels of university students (Low/Moderate vs. High) based on demographic, academic, and psychological survey data.

By explicitly avoiding high-level Machine Learning libraries (like TensorFlow or PyTorch), this project demonstrates a deep, mathematical understanding of forward propagation, backpropagation, and advanced optimization algorithms.

## Tech Stack
* **Language:** Python
* **Libraries:** `NumPy`, `Pandas`, `Matplotlib` / `Seaborn`

## Model Architecture & Advanced Features
The custom Neural Network class includes several production-ready ML techniques:
* **Optimizers:** Implementation of SGD, Momentum, RMSprop, and Adam optimizers.
* **Overfitting Prevention:** Integrated L2 Regularization, Dropout layers, and Early Stopping.
* **Stability:** Built-in Batch Normalization to stabilize and accelerate training.
* **Ensemble Learning:** A custom Ensemble class that aggregates predictions from 10 independent networks to reduce variance and improve model robustness.

## Data Preprocessing
To ensure model reliability and prevent data leakage, the following preprocessing steps were applied:
* **Imputation:** Missing numerical values were replaced with column medians.
* **Normalization:** Min-Max scaling was applied to bring all features into a [0,1] range.
* **Class Balancing:** Random oversampling was used on the training set to prevent the network from developing a majority-class bias.
* **Feature Selection:** 10 survey questions directly measuring stress (and the original `Stress Value`) were dropped to force the model to infer stress from indirect indicators.

## Results & Findings
Extensive hyperparameter tuning was conducted (averaging results over 5 independent runs):
* **Activation Functions:** The `ReLU` activation function outperformed `Sigmoid` and `Tanh`, preventing vanishing gradients in deeper layers.
* **Overfitting Control:** The combination of Dropout and L2 regularization successfully forced the network to learn generalized patterns rather than memorizing the training data.
* **Ensemble vs. Single Model:** The Ensemble model (10 networks) achieved a more stable and higher performance (Accuracy: ~80.5%, F1-Macro: ~0.751) compared to a single network architecture.

## Documentation
For a detailed breakdown of the experiments, learning curves, and confusion matrices, please refer to the `Raport.docx` file included in this repository.
