import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house_data.csv')

# Handle categorical variables by one-hot encoding
data = pd.get_dummies(data, columns=['Location', 'Zip_Code'])

# Selecting features and target variable
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Create a DataFrame with actual prices and predicted prices
results = pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': predictions})

# Save the results to a CSV file
results.to_csv('predicted_prices.csv', index=False)

# Scatter plot for actual prices vs. predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Actual Prices vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)  # Diagonal line showing perfect prediction
plt.show()
