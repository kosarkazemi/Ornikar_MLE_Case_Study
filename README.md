
# Ornikar_MLE_Case_Study

## Overview
Test ML Software Engineer for Akeneo.
The aim of this project is to predict the lead client conversions based on their informations and behaivoirs.

## Lead Conversion PredictionAPI Documentation

Welcome to the Lead Conversion Prediction API! This API provides a prediction of clients conversion based on their input data.
It utilizes Flask, a popular Python web framework, to handle incoming HTTP requests.


### Getting Started

#### Installation Instructions

1. **Install Docker:**
   Ensure you have Docker installed on your system. Docker is a containerization platform that simplifies deployment and dependencies.
   - [Docker Installation Guide](https://docs.docker.com/get-docker/)

2. **Clone the Repository:**
   Clone the repository containing the API code to your local machine.
   ```bash
   git clone https://github.com/kosarkazemi/Ornikar_MLE_Case_Study.git
   cd Ornikar_MLE_Case_Study
   ```

3. **Install Poetry:**
   Poetry is a tool for dependency management and packaging in Python. It allows you to manage project dependencies and create a virtual environment.
   - [Poetry Installation Guide](https://python-poetry.org/docs/)

4. **Install Dependencies:**
   Use Poetry to install the project dependencies.
   ```bash
   poetry install
   ```

5. **Build and Run the Docker Image:**
   Build the Docker image from the provided Dockerfile and run the API inside a Docker container.
   ```bash
   docker build -t ornikar_mle .
   docker run -p 5001:5000 ornikar_mle
   ```
   The API will now be running and accessible at `http://0.0.0.0:5001`.

### API Endpoints

1. **`GET /`**

   This endpoint provides a welcome message to users accessing the root URL of the API.
   
   **Request:**
   ```plaintext
   GET /
   ```

   **Response:**
   ```json
   {
     "message": "Welcome to the lead conversion prediction API!"
   }
   ```

2. **`POST /predict_lead_conversion`**

   This endpoint predicts lead conversion based on input data.

   **Request:**
   - Method: `POST`
   - Body:
     ```json
     {
       "model_name": "your_model_name",
       "input_data": "your_input_data"
     }
     ```

   **Response:**
   ```json
   {
     "predictions": [0.75]
   }
   ```

### Usage

- **Access the API:**
  Use a tool like Postman or send HTTP requests programmatically to the specified endpoints.

- **Run the client.py Script:**
  Execute the client.py script by running it with Python. This script will send a POST request to the API, predict lead conversion, and display the response. Use the following command:
  ```bash
  poetry run python client.py
  ```
  This script will display the payload being sent, the response status code, and the predicted lead conversion probability.



## Deployment to the Cloud (Future Enhancement)

To deploy this API in a cloud architecture for production use, consider the following steps:


1. **Data Ingestion:**
   Employ ETL (Extract, Transform, Load) processes to extract data from a data warehouse, transform it for training, and load it into a storage system, such as cloud storage solutions like Amazon S3 or Google Cloud Storage. This can be achieved through technologies like Apache Airflow, AWS Glue, or custom scripts.

2. **Model Training and Updating:**
   Utilize cloud-based services to enable distributed training of large models, all while automating the process of training models using freshly ingested data. This can be accomplished with the assistance of technologies such as AWS SageMaker, Google AI Platform, CI/CD pipelines, Kubeflow, or custom scripts.

3. **Model Versioning:**
   Serialize and version trained models for easy deployment and rollback by leveraging technologies like MLflow, Kubeflow, or custom solutions.

4. **Model Deployment and API Gateway:**
   Deploy the model as an API, Docker container, or serverless function on the cloud, using technologies such as AWS Lambda, Google Cloud Run, or Kubernetes. Expose the model through gateway APIs for simplified consumption, employing technologies like AWS API Gateway, Google Cloud Endpoints, or custom REST APIs. For enhanced flexibility and scalability, ensure that the deployment runs within Docker containers.

5. **Update Trigger:**
   Implement a trigger mechanism designed to detect new data in the data warehouse, with serverless functions like AWS Lambda serving as viable technology options.


By following these steps, you can deploy the API in a cloud architecture, ensuring scalability, regular updates, and fast predictions for the Tech team. Be sure to monitor the performance and costs to optimize resource usage.
```