import pandas as pd
import joblib
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import train_model
from src.serving import predict
from sklearn.metrics import accuracy_score, classification_report
import config

if __name__ == '__main__':
    # Specify the path to the data file
    data_file_path = 'data/long_quotes.csv'

    # Load the data
    data = load_data(data_file_path)

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)

    # Train the machine learning model
    model = train_model(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, config.model_file_path)

    # Predict using the trained model
    y_pred = predict(model, X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Report the results
    print(f"Model accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_rep)

    # You can also save the evaluation results to a file if needed
    with open('results/evaluation.txt', 'w') as f:
        f.write(f"Model accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(classification_rep)
