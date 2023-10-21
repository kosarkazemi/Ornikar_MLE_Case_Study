import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import re

def load_data(file_path):
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def extract_numeric(text):
    """
    Extract numeric values from a text string.
    Args:
        text (str): Input text.
    Returns:
        float: Extracted numeric value.
    """
    if (not text) or (isinstance(text, float)):
        return np.nan
    match = re.match(r'(\d+)(?:-(\d+))?(\+)?', text)
    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2)) if match.group(2) else num1
        is_plus = match.group(3)
        if is_plus:
            return num1
        else:
            return (num1 + num2) / 2

def convert_timestamps_to_datetime(data):
    """
    Convert timestamp columns to datetime objects.
    Args:
        data (pd.DataFrame): Input data.
    """
    data['submitted_at'] = pd.to_datetime(data['submitted_at']).dt.tz_convert('UTC')
    data['effective_start_date'] = pd.to_datetime(data['effective_start_date'])
    data['effective_start_date'] = data['effective_start_date'].dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    data['effective_start_date'] = pd.to_datetime(data['effective_start_date'])

def calculate_time_difference(data):
    """
    Calculate the time difference in days.
    Args:
        data (pd.DataFrame): Input data.
    """
    data['time_difference_days'] = (data['effective_start_date'] - data['submitted_at']).dt.days

def drop_unnecessary_columns(data):
    """
    Drop unnecessary or sparse columns.
    Args:
        data (pd.DataFrame): Input data.
    """
    columns_to_drop = ['last_utm_source', 'long_quote_id', 'lead_id', 'rbs_result', 'policy_subscribed_at',
                'contract_id', 'payment_frequency', 'submitted_at', 'effective_start_date', 'has_subscribed_online']
    data.drop(columns_to_drop, axis=1, inplace=True)

def map_intensity_values(data):
    """
    Map intensity values to numerical values.
    Args:
        data (pd.DataFrame): Input data.
    """
    mapping_dict_intensity = {'low': 1, 'medium': 2, 'high': 3, np.nan: 0}
    columns_to_map = ['annual_price_third_party', 'annual_price_intermediate', 'annual_price_all_risks']
    for column in columns_to_map:
        data[column] = data[column].replace(mapping_dict_intensity)

def encode_categorical_features(data):
    """
    Encode categorical features.
    Args:
        data (pd.DataFrame): Input data.
    """
    encoder = LabelEncoder()
    categorical_cols = ['provider', 'product_third_party', 'product_intermediate', 'product_all_risks',
                        'chosen_formula', 'chosen_product', 'main_driver_gender', 'vehicle_class', 'vehicle_region']
    for col in categorical_cols:
        data[col] = data[col].fillna('unknown')
        data[col] = encoder.fit_transform(data[col])

def standardize_numerical_features(data):
    """
    Standardize numerical features.
    Args:
        data (pd.DataFrame): Input data.
    """
    numerical_cols = ['main_driver_age', 'main_driver_licence_age', 'main_driver_bonus', 'vehicle_age', 'vehicle_group']
    data[numerical_cols] = data[numerical_cols].applymap(extract_numeric)
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

def preprocess_data(data):
    """
    Preprocess the input data.
    Args:
        data (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    preprocessed_data = data.copy()
    convert_timestamps_to_datetime(preprocessed_data)
    calculate_time_difference(preprocessed_data)
    drop_unnecessary_columns(preprocessed_data)
    encode_categorical_features(preprocessed_data)
    standardize_numerical_features(preprocessed_data)
    map_intensity_values(preprocessed_data)
    preprocessed_data.dropna(inplace=True)
    preprocessed_data.to_csv('data/preprocessed_data.csv', index=False)
    print("Finished with preprocessing")
    return preprocessed_data

def split_data(data):
    """
    Split data into features and target.
    Args:
        data (pd.DataFrame): Input data.
    Returns:
        pd.DataFrame: Features (X_train, X_test) and target (y_train, y_test).
    """
    X = data.drop('has_subscribed', axis=1)
    y = data['has_subscribed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test.to_csv('data/X_test.csv', index=False)
    return X_train, X_test, y_train, y_test

def preprocess_input_data(input_data):
    """
    Preprocess input data to make it suitable for predictions.
    Args:
        input_data (dict): Input data for predictions.
    Returns:
        pd.DataFrame: Preprocessed input data.
    """
    # Create a DataFrame from the input data
    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
    # Perform the same preprocessing steps as in preprocess_data
    convert_timestamps_to_datetime(input_df)
    calculate_time_difference(input_df)
    drop_unnecessary_columns(input_df)
    encode_categorical_features(input_df)
    standardize_numerical_features(input_df)
    map_intensity_values(input_df)
    input_df.dropna(inplace=True)

    return input_df

if __name__ == '__main__':
    data_file_path = 'data/long_quotes.csv'
    data = load_data(data_file_path)
    preprocessed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
