from src.data_preprocessing import preprocess_input_data

def predict(model, input_data, do_prepocess=True):
    """
    Predict using a machine learning model.
    Args:
        model: Machine learning model (e.g., scikit-learn model).
        input_data (dict): Input data for predictions.
    Returns:
        list: Predictions.
    """
    if do_prepocess:
        # Preprocess the input data
        preprocessed_input_data = preprocess_input_data(input_data)
        predictions = model.predict(preprocessed_input_data)
    else:
        predictions = model.predict(input_data)

    return predictions