
# Iris Flower Classification using Decision Tree and Pickle

This project demonstrates a simple machine learning workflow using the **Iris dataset** and a **Decision Tree Classifier**. The model is trained on the dataset, and both the trained model and test data are saved as pickle files for later use and evaluation.

## Files in this Repository

1. **iris_pickle.py**: This Python script handles the complete process of data loading, training the Decision Tree model, saving the trained model and test data, and evaluating the model.
2. **iris_model.pkl**: The trained Decision Tree model, serialized using pickle for reuse without retraining.
3. **iris_test_data.pkl**: The test dataset, containing the test features and labels used to evaluate the model after it is reloaded.
4. **requirements_iris.txt**: This file lists the dependencies required to run the Python script, including libraries like `scikit-learn` for machine learning and `pandas` for data manipulation.

### Workflow Explanation

1. **Data Handling**: The script loads the famous Iris dataset, which contains information about iris flowers, classified into three species based on features like sepal and petal dimensions.
2. **Model Training**: A Decision Tree Classifier is trained using a portion of the dataset, while the remainder is reserved for testing.
3. **Pickle for Model Persistence**: After training, the model is saved in a serialized format (`iris_model.pkl`) using pickle, allowing it to be reused without retraining.
4. **Test Data Storage**: The test data (features and labels) is also saved in a separate pickle file (`iris_test_data.pkl`) for future evaluations.
5. **Model Evaluation**: The script demonstrates how the saved model can be loaded from the pickle file and evaluated on the test data. The evaluation is done by comparing the predicted labels against the true labels, calculating metrics such as accuracy.

This project highlights the use of **model serialization** with pickle, enabling efficient model storage and reuse, which is crucial for production environments where retraining models every time is inefficient.
