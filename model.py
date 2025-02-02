import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
data = pd.read_csv("data.csv")

# Prepare features and target
X = data[['Avg.Area Income', 'Avg.Area House Age', 'Avg.Area Number of Rooms',
          'Avg.Area Number of Bedrooms', 'Area Population']]
y = data['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open("house_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as house_price_model.pkl")