import requests
import json
from src.data_preprocessing import load_data
import config

# Define the API endpoint URL
host = '0.0.0.0'
api_url = f'http://{host}:5001/predict_lead_conversion'  # Update with your API URL

# Create input data for making predictions
df_data = load_data(config.data_file)
df_test_data = load_data(config.test_data_file)
df_test_data.rename(columns={df_test_data.columns[0]: "index" }, inplace = True)

# remove the result column
input_data = df_data.loc[df_test_data[df_test_data.columns[0]]].sample(n=1)
input_data.drop(["has_subscribed"], axis=1, inplace=True)
input_data_dict = input_data.to_dict(orient='records')[0]

# Create the payload dictionary
payload = {
    'input_data': input_data_dict,
    'model_name': "trained_model.pkl"
}

# Print the payload for verification
print(json.dumps(payload, indent=4))

# Convert the payload to JSON
payload_json = json.dumps(payload)

# Send a POST request to the API
response = requests.post(api_url, json=payload_json)

# Check if the request was successful
if response.status_code == 200:
    # Print the response
    result = response.json()
    print("Predictions:")
    print(result['predictions'])
else:
    print("Error:", response.status_code, response.text)
