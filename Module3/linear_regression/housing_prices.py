import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Load the California Housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
# Add PRICE at the end
data['PRICE'] = housing.target

# Feature engineering
data['RoomPerPerson'] = data['AveRooms'] / data['AveOccup']
data['BedroomPerRoom'] = data['AveBedrms'] / data['AveRooms']

# Reorder columns to ensure PRICE is last
cols = [col for col in data.columns if col != 'PRICE'] + ['PRICE']
data = data[cols]

# Create correlation matrix visualization
plot.figure(figsize=(12, 8))
correlation_matrix = data.corr().round(2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plot.title('Feature Correlation Matrix')
plot.tight_layout()
plot.show()

# Print correlations with price
print("\nCorrelations with Housing Price:")
print("-------------------------------")
correlations = correlation_matrix['PRICE'].sort_values(ascending=False)
print(correlations)

# Select features with stronger correlations (|correlation| > 0.15)
correlations = correlations.abs()
selected_features = correlations[correlations > 0.15].index.tolist()
selected_features.remove('PRICE')  # Remove target variable from features

print("\nSelected Features:")
print("-----------------")
print(selected_features)

# Create a copy of the selected features
X = data[selected_features].copy()
y = data['PRICE']

# Add polynomial feature for MedInc using loc
X.loc[:, 'MedInc^2'] = X['MedInc'] ** 2

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\nCross-validation R² scores:", cv_scores)
print("Mean CV R² score: {:.2f} (+/- {:.2f})".format(cv_scores.mean(), cv_scores.std() * 2))

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance:")
print("-----------------")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R2 Score: {train_r2:.2f}")
print(f"Testing R2 Score: {test_r2:.2f}")

# Create visualization for actual vs predicted values
plot.figure(figsize=(12, 5))

# Training set
plot.subplot(1, 2, 1)
plot.scatter(y_train, y_train_pred, alpha=0.5, color='blue')
plot.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plot.xlabel('Actual Price')
plot.ylabel('Predicted Price')
plot.title('Training Set: Actual vs Predicted Prices')

# Testing set
plot.subplot(1, 2, 2)
plot.scatter(y_test, y_test_pred, alpha=0.5, color='green')
plot.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plot.xlabel('Actual Price')
plot.ylabel('Predicted Price')
plot.title('Testing Set: Actual vs Predicted Prices')

plot.tight_layout()
plot.show()

# Feature importance visualization
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
coefficients['AbsCoefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('AbsCoefficient', ascending=True)

plot.figure(figsize=(10, 6))
sns.barplot(data=coefficients, x='Feature', y='Coefficient', hue='Feature', legend=False)
plot.title('Feature Coefficients')
plot.xticks(rotation=45)
plot.tight_layout()
plot.show()

# Print feature importance
print("\nFeature Importance:")
print("------------------")
for _, row in coefficients.iloc[::-1].iterrows():
    print(f"{row['Feature']}: {row['Coefficient']:.4f}")