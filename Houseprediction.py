import numpy as np
from sklearn.linear_model import LinearRegression

# Example dataset (size in sq. feet, number of bedrooms, price in thousands)
X = np.array([[1400, 3], [1600, 3], [1700, 4], [1875, 3], [1100, 2]])
y = np.array([245, 312, 279, 308, 199])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a house with 1500 sq feet and 3 bedrooms
predicted_price = model.predict([[1500, 3]])
print(f"Predicted house price: ${predicted_price[0] * 1}")