import pickle
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_iris_data():
    """Load the Iris dataset and split into train and test sets."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Iris dataset loaded and split into train/test successfully.")
    return X_train, X_test, y_train, y_test

def train_and_save_model(X_train, y_train):
    """Train a Decision Tree Classifier and save the model to a pickle file."""
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    logging.info("Model trained successfully.")

    # Save the model to a pickle file
    with open("iris_model.pkl", "wb") as file:
        pickle.dump(model, file)
        logging.info("Model saved to iris_model.pkl")

def load_pickle_file(file_path):
    """Load a pickle file."""
    with open(file_path, "rb") as file:
        data = pickle.load(file)
        logging.info(f"Loaded {file_path} successfully.")
    return data

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

# Main execution: Train the model and then evaluate it
X_train, X_test, y_train, y_test = load_iris_data()
train_and_save_model(X_train, y_train)

# Save test data to a pickle file for later evaluation
test_data = {'X_test': X_test, 'y_test': y_test}
with open("iris_test_data.pkl", "wb") as file:
    pickle.dump(test_data, file)
    logging.info("Test data saved to iris_test_data.pkl")

# Load the model and test data for evaluation
model = load_pickle_file("iris_model.pkl")
test_data = load_pickle_file("iris_test_data.pkl")

# Extract test data
X_test = test_data['X_test']
y_test = test_data['y_test']

# Evaluate the model
evaluate_model(model, X_test, y_test)
