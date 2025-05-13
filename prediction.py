import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Loads the model parameters theta0 and theta1 from a JSON file
def load_model(file_path):
    with open(file_path, 'r') as file:
        model = json.load(file)
    return model['theta0'], model['theta1']

# Reads the dataset from a CSV file and returns mileage and price arrays
def read_dataset(file_path):
    data = pd.read_csv(file_path)
    return data['km'].values, data['price'].values

# Predicts the price based on the given mileage using the model parameters
def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

# Calculates the Mean Squared Error (MSE)
def calculate_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)

# Calculates the R-squared (R²) value
def calculate_r_squared(actual, predicted):
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    return 1 - (ss_residual / ss_total)

# Calculates the composite precision score
def calculate_precision(mse, r_squared, max_mse):
    normalized_mse = mse / max_mse
    precision_score = (1 - normalized_mse + r_squared) / 2
    return precision_score

# Plots the results and displays the plot in a window with a specified size
def plot_results(mileage, price, theta0, theta1, input_mileage, estimated_price, mse, r_squared, precision_score):
    plt.figure(figsize=(14, 10))  # Set the figure size to 14 inches by 10 inches
    plt.scatter(mileage, price, color='blue', label='Actual Data')
    
    # Generating points for the regression line
    regression_line = theta0 + theta1 * mileage
    plt.plot(mileage, regression_line, color='red', label='Regression Line')
    
    # Adding the estimated price point
    plt.scatter([input_mileage], [estimated_price], color='green', label='Estimated Price', zorder=5)
    
    # Adding labels, title, and legend
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Linear Regression: Mileage vs Price')
    plt.legend()
    
    # Adding precision metrics as text annotations
    plt.text(0.05, 0.05, f'Mean Squared Error: {mse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    plt.text(0.05, 0.10, f'R²: {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    plt.text(0.05, 0.15, f'Precision: {precision_score * 100:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom')
    
    plt.show()

def main():
    theta0, theta1 = load_model('model.json')
    mileage, price = read_dataset('data.csv')
    
    predicted_prices = predict_price(mileage, theta0, theta1)
    mse = calculate_mse(price, predicted_prices)
    r_squared = calculate_r_squared(price, predicted_prices)
    
    max_mse = calculate_mse(price, np.full_like(price, np.mean(price)))
    
    precision_score = calculate_precision(mse, r_squared, max_mse)
    
    input_mileage = float(input("Enter the mileage: "))
    estimated_price = predict_price(input_mileage, theta0, theta1)
    print(f"Estimated price for mileage {input_mileage}: {estimated_price}")
    
    # Plot results with precision metrics and the estimated price point
    plot_results(mileage, price, theta0, theta1, input_mileage, estimated_price, mse, r_squared, precision_score)

if __name__ == "__main__":
    main()