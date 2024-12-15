import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Tuple, Union

class KNNClassifier:
    def __init__(self, k: int = 5):
        """
        Initialize KNN classifier
        
        Args:
            k (int): Number of neighbors to use for classification
        """
        # Step 2: Initialize the value of k
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def load_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare the data
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        # Step 1: Load the data
        data = pd.read_csv(file_path, header=None)
        
        # Separate features and labels
        X = data.iloc[:, :4].values  # First 4 columns are features
        y = data.iloc[:, 4].values   # Last column is the label (genre)
        
        return X, y
    
    def euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            point1 (np.ndarray): First point
            point2 (np.ndarray): Second point
            
        Returns:
            float: Euclidean distance between the points
        """
        # Step 4: Calculate the distance between test data and each row of training data
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier with training data
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict classes for test data
        
        Args:
            X (np.ndarray): Test features
            
        Returns:
            List[str]: Predicted classes
        """
        predictions = []
        
        # Step 3: For getting the predicted class, iterate from 1 to total number of training data points
        for test_point in X:
            # Calculate distances between test point and all training points
            distances = []
            for idx, train_point in enumerate(self.X_train):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[idx]))
            
            # Step 5: Sort the calculated distances in ascending order based on distance values
            distances.sort(key=lambda x: x[0])
            
            # Step 6: Get top k rows from the sorted array
            k_nearest = distances[:self.k]
            
            # Step 7: Get the most frequent class of these rows
            k_nearest_labels = [label for _, label in k_nearest]
            most_common = Counter(k_nearest_labels).most_common(1)
            
            # Step 8: Return the predicted class
            predictions.append(most_common[0][0])
            
        return predictions

if __name__ == "__main__":
    # Initialize classifier
    knn = KNNClassifier(k=5)
    
    # Load and split data
    X, y = knn.load_data("data.csv")
    
    # Simple train-test split
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Fit and predict
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = sum(pred == true for pred, true in zip(predictions, y_test)) / len(y_test)
    print(f"Accuracy: {accuracy:.2f}")