# Linear_Regression - Introduction to Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An implementation of linear regression with gradient descent to predict car prices based on mileage.

## 📖 Project Summary

This project implements a basic machine learning algorithm:
> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." - Tom M. Mitchell

**Key Features:**
- Predicts car prices using single-variable linear regression
- Implements gradient descent training from scratch
- Includes visualization of results (bonus)
- Follows strict academic requirements (no mathematical libraries used)

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
# Clone this repository
git clone https://github.com/Dieau/Linear_Regression.git
cd ft_linear_regression

# Run the installation script
./setup.sh
```

The setup script will:

   - Create a Python virtual environment

   - Install all dependencies

   - Install the project in editable mode

## 🚀 Usage

1. **First Prediction Attempt (Untrained Model)**

         python prediction.py


   You'll be prompted to enter a car mileage

   Since the model isn't trained yet, predictions will be inaccurate (θ₀ and θ₁ are defaulted to 0).

   Example output:

         Enter mileage: 50000
         Estimated price for mileage 50000: 0 (model not trained)


2. **Train the Model**

         python training.py

    Processes the dataset using gradient descent

    Saves trained parameters (θ₀ and θ₁) to model.json

    Example output:

         {"theta0": 8480.966554681116, "theta1": -0.021271757648460635}

3. **Make Accurate Predictions**

         python prediction.py

    Now uses the trained parameters from thetas.json

    Example output:

         Enter the mileage: 82029
         Estimated price for mileage 82029.0: 6736.0655465355385

## 🔑 Key Concepts Explained

### Core Regression Components
- **θ₀ (Theta0)**: The y-intercept of our regression line (base price when mileage is 0)
- **θ₁ (Theta1)**: The slope of our regression line (price change per unit mileage)
- **Regression Line**: The straight line (`price = θ₀ + θ₁×mileage`) that best fits our data points

### Data Processing
- **Normalization**: Scaling mileage/price to [0,1] range to equalize feature influence
- **Denormalization**: Converting θ parameters back to original data scale after training

### Training Mechanics
- **Gradient**: The direction and rate of steepest cost function increase
- **Gradient Clipping**: Limiting gradient updates to prevent overshooting (threshold=1.0)
- **Learning Rate (0.01)**: Step size for parameter updates during gradient descent

### Performance Metrics
- **Mean Squared Error (MSE)**: Average squared difference between predicted/actual prices
- **R-squared (R²)**: Proportion of price variance explained by mileage (0-1 scale)
- **Composite Precision**: Combined score balancing MSE and R² (0% = random, 100% = perfect)

### Mathematical Foundations

```math
\text{Gradient Descent Update:}
\begin{cases}
θ₀ &= θ₀ - α\frac{1}{m}\sum(\text{predicted} - \text{actual}) \\
θ₁ &= θ₁ - α\frac{1}{m}\sum((\text{predicted} - \text{actual})×\text{mileage})
\end{cases}
```

Where:

    α = learning rate

    m = number of data points
