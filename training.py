import pandas as pd
import numpy as np
import json
import matplotlib
from termcolor import colored
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#Parse dataset csv file
def read_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['km'].values, data['price'].values

# Normalizes the mileage and price data to a range of [0, 1]
def normalize_data(mileage, price):
    mileage_min, mileage_max = mileage.min(), mileage.max()
    price_min, price_max = price.min(), price.max()
    mileage_normalized = (mileage - mileage_min) / (mileage_max - mileage_min)
    price_normalized = (price - price_min) / (price_max - price_min)
    return mileage_normalized, price_normalized, mileage_min, mileage_max, price_min, price_max

# Computes the gradients of the cost function with respect to theta0 and theta1
def compute_gradients(mileage, price, theta0, theta1):
    m = len(mileage)
    error_sum_0 = 0
    error_sum_1 = 0
    for i in range(m):
        error = (theta0 + theta1 * mileage[i]) - price[i]
        error_sum_0 += error
        error_sum_1 += error * mileage[i]
    gradient_theta0 = (1 / m) * error_sum_0
    gradient_theta1 = (1 / m) * error_sum_1
    return gradient_theta0, gradient_theta1

# Clips the gradient to a specified threshold to prevent large updates
def clip_gradient(gradient, threshold):
    if gradient > threshold:
        return threshold
    elif gradient < -threshold:
        return -threshold
    return gradient

# Trains the linear regression model using gradient descent
def train_model(mileage, price, learning_rate, iterations, gradient_threshold):
    theta0, theta1 = 0, 0
    for _ in range(iterations):
        gradient_theta0, gradient_theta1 = compute_gradients(mileage, price, theta0, theta1)
        gradient_theta0 = clip_gradient(gradient_theta0, gradient_threshold)
        gradient_theta1 = clip_gradient(gradient_theta1, gradient_threshold)
        theta0 -= learning_rate * gradient_theta0
        theta1 -= learning_rate * gradient_theta1
    return theta0, theta1

# Denormalize theta0 and theta1 to their original scale
def denormalize_theta(theta0, theta1, mileage_min, mileage_max, price_min, price_max):
    theta1 = theta1 * (price_max - price_min) / (mileage_max - mileage_min)
    theta0 = price_min + theta0 * (price_max - price_min) - theta1 * mileage_min
    return theta0, theta1

# Save the model parameters theta0 and theta1 to a JSON file
def save_model(theta0, theta1, file_path):
    model = {'theta0': theta0, 'theta1': theta1}
    with open(file_path, 'w') as file:
        json.dump(model, file)

# Plot the data points and regression line
def plot_results(mileage, price, theta0, theta1):
    plt.figure(figsize=(14, 10))
    plt.scatter(mileage, price, color='blue', label='Actual Data')
    regression_line = theta0 + theta1 * mileage
    plt.plot(mileage, regression_line, color='red', label='Regression Line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression: Mileage vs Price')
    plt.legend()
    plt.show()

def main():
    mileage, price = read_dataset('data.csv')
    mileage_normalized, price_normalized, mileage_min, mileage_max, price_min, price_max = normalize_data(mileage, price)
    theta0, theta1 = train_model(mileage_normalized, price_normalized, learning_rate=0.01, iterations=10000, gradient_threshold=1.0)
    theta0, theta1 = denormalize_theta(theta0, theta1, mileage_min, mileage_max, price_min, price_max)
    save_model(theta0, theta1, 'model.json')
    print(colored("Model trained successfully. Model parameters saved to 'model.json'.", 'green'))
    print(colored(f"Theta0: {theta0}, Theta1: {theta1}", 'cyan'))
    print(colored("Iterations: 10000, Learning Rate: 0.01, Gradient Threshold: 1.0", 'cyan'))
    plot_results(mileage, price, theta0, theta1)

if __name__ == "__main__":
    main()