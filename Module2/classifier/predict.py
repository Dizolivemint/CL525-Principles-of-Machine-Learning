from knn_classifier import KNNClassifier
import numpy as np

def validate_input(value: str, min_val: float, max_val: float, field_name: str) -> float:
    """
    Validates input to ensure it's a float within acceptable ranges.
    
    Args:
        value (str): The input string to validate
        min_val (float): Minimum acceptable value
        max_val (float): Maximum acceptable value
        field_name (str): Name of the field being validated
        
    Returns:
        float: The validated float value
    
    Raises:
        ValueError: If input is invalid
    """
    try:
        float_value = float(value)
        if not (min_val <= float_value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
        return float_value
    except ValueError as e:
        if str(e).startswith(field_name):
            raise
        raise ValueError(f"Invalid input: {field_name} must be a number")

def get_demographic_input() -> np.ndarray:
    """
    Gets and validates demographic input from user.
    
    Returns:
        np.ndarray: Array containing [age, height, weight, gender]
    """
    while True:
        try:
            # Get age
            age = validate_input(
                input("Enter age (in years): "),
                min_val=0,
                max_val=120,
                field_name="Age"
            )
            
            # Get height
            height = validate_input(
                input("Enter height (in inches): "),
                min_val=20,
                max_val=108,
                field_name="Height"
            )
            
            # Get weight
            weight = validate_input(
                input("Enter weight (in pounds): "),
                min_val=1,
                max_val=1000,
                field_name="Weight"
            )
            
            # Get gender
            gender = validate_input(
                input("Enter gender (0 for female, 1 for male): "),
                min_val=0,
                max_val=1,
                field_name="Gender"
            )
            
            return np.array([[age, height, weight, gender]])
            
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again.\n")

def main():
    """Main function to run the program."""
    try:
        # Initialize KNN classifier
        knn = KNNClassifier(k=5)
        
        # Load training data
        X, y = knn.load_data("data.csv")
        
        # Train the classifier
        knn.fit(X, y)
        
        print("\nPlease enter demographic information:")
        print("------------------------------------")
        
        # Get input from user
        input_data = get_demographic_input()
        
        # Make prediction
        prediction = knn.predict(input_data)[0]  # Get first prediction since we only have one input
        
        print("\nInput Summary:")
        print(f"Age: {input_data[0,0]:.1f} years")
        print(f"Height: {input_data[0,1]:.1f} inches")
        print(f"Weight: {input_data[0,2]:.1f} lbs")
        print(f"Gender: {'Male' if input_data[0,3] == 1 else 'Female'}")
        print(f"\nPredicted Category: {prediction}")
        
    except FileNotFoundError:
        print("Error: Could not find data.csv file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()