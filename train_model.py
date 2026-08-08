import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error


# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

#Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train the model
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train) # Train the model on the training data
print("Model trained successfully.")

#Prediction
y_pred = model.predict(X_test)
print("Predictions made successfully.")



#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#Save the model
joblib.dump(model, 'housing_regression_model.joblib') # Save the model to a file
print("Model saved successfully.")