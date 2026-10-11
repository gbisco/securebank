# SecureBank Fraud Detection System

## Overview
SecureBank is a machine learning system prototype for detecting fraudulent financial transactions. The goal of this project is to turn messy, multi-source transaction data into a working fraud detection service that can be trained, deployed, monitored, and tested through a simple API.

The system supports three main workflows: generating a cleaned training dataset, training and saving a fraud detection model, and predicting whether a new transaction is legitimate or fraudulent. It uses both transaction data and customer profile data, then packages the full workflow inside a Dockerized Flask application.

At a high level, the project combines data preprocessing, feature engineering, model training, threshold selection, prediction serving, and JSON logging. The final result is not just a model notebook, but a small deployable ML system built around reproducible scripts and API endpoints.

## System Capabilities
The SecureBank system supports the core workflows needed to build and operate a fraud detection prototype. It can generate a cleaned training dataset from the raw data sources, train a versioned model, serve predictions through an API, and write monitoring logs for system activity.

The system supports both logistic regression and gradient boosting workflows. Logistic regression was used as a baseline model during experimentation, while gradient boosting is configured as the default model because it produced stronger fraud detection performance. In the latest smoke-tested training run, the default gradient boosting configuration selected a `0.99` threshold and produced approximately `0.865` precision, `0.666` recall, and `0.753` F1 on the held-out test split.

| Capability             | Implementation                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| Fraud prediction       | `POST /predict` accepts a transaction JSON payload and returns `legitimate` or `fraudulent`. |
| Dataset generation     | `POST /create_dataset` runs the preprocessing pipeline and saves a cleaned dataset.          |
| Model training         | `POST /train_model` trains the default fraud detection model and evaluates thresholds.       |
| Model versioning       | Trained models and metadata are saved under `securebank/storage/models/`.                    |
| Monitoring and logging | API activity is logged as timestamped JSON files under `securebank/logs/`.                   |
| Docker deployment      | The system runs inside a Docker container using `python:3.9-slim`.                           |


## Project Structure

```text
securebank-gbiscor1/
├── .dockerignore
├── .gitignore
├── analysis/
│   ├── data_analysis.ipynb
│   ├── raw_data_analysis.ipynb
│   └── model_efficientcy.ipynb
├── securebank/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── test.json
│   ├── data_sources/
│   ├── executables/
│   │   ├── run_server.sh
│   │   ├── create_dataset.sh
│   │   ├── train_model.sh
│   │   └── predict.sh
│   ├── logs/
│   ├── storage/
│   │   ├── datasets/
│   │   └── models/
│   └── modules/
│       ├── data/
│       ├── evaluation/
│       ├── log/
│       ├── models/
│       ├── predict/
│       ├── services/
│       └── train/
└── test/
    ├── fixtures/
    ├── modules/
    │   ├── data/
    │   ├── evaluation/
    │   ├── log/
    │   ├── models/
    │   ├── predict/
    │   ├── services/
    │   └── train/
    ├── grade_runner.py
    ├── unit_test_2.py
    └── unit_test.py
```

### Architecture Notes

The repository is organized around the main runtime package in `securebank/`, supporting analysis notebooks in `analysis/`, and validation files in `test/`. The application follows a layered structure: the Flask app exposes endpoints, services handle endpoint-specific behavior, orchestrators coordinate training and prediction workflows, and model/data modules handle the lower-level implementation details.

#### Root Files

| Path | Purpose |
|---|---|
| `.dockerignore` | Controls which files are excluded from the Docker build context. This keeps the image smaller and avoids copying unnecessary local artifacts. |
| `.gitignore` | Prevents raw data, generated models, datasets, logs, cache files, and local environment files from being committed. |

#### `analysis/`

| File | Purpose |
|---|---|
| `raw_data_analysis.ipynb` | Inspects the original customer, transaction, and fraud data sources before preprocessing. |
| `data_analysis.ipynb` | Analyzes the cleaned dataset and explores fraud relationships with transaction, customer, category, and time-based variables. |
| `model_efficientcy.ipynb` | Contains model experimentation and comparison work used to evaluate practical model performance. |

#### `securebank/`

| File or Directory | Purpose |
|---|---|
| `app.py` | Main Flask application. It exposes `/`, `/create_dataset`, `/train_model`, and `/predict`, then delegates work to the service layer. |
| `Dockerfile` | Builds the Docker image using `python:3.9-slim`, installs requirements, exposes port `5000`, and starts the Flask app. |
| `requirements.txt` | Lists the Python dependencies needed by the API, preprocessing pipeline, model training, and prediction workflows. |
| `test.json` | Example transaction payload used by `predict.sh` to test the `/predict` endpoint. |
| `data_sources/` | Expected location for the raw SecureBank data files. These files are required at runtime but should not be committed to Git. |
| `executables/` | Contains bash scripts used to start the server and test the required API endpoints. |
| `logs/` | Stores timestamped JSON monitoring logs generated by preprocessing, dataset creation, training, and prediction workflows. |
| `storage/` | Stores generated datasets, trained models, model metadata, and the default model pointer. |
| `modules/` | Contains the core implementation for data processing, model training, prediction, evaluation, logging, and service orchestration. |

#### `securebank/executables/`

| Script | Purpose |
|---|---|
| `run_server.sh` | Builds the Docker image and runs the SecureBank API container on the exposed port. |
| `create_dataset.sh` | Sends a request to `/create_dataset` to generate and save a cleaned training dataset. |
| `train_model.sh` | Sends a request to `/train_model` to train the configured model, evaluate thresholds, save metadata, and register the default model. |
| `predict.sh` | Sends the example `test.json` transaction payload to `/predict` and returns a fraud prediction. |

#### `securebank/storage/`

| Directory or File | Purpose |
|---|---|
| `storage/datasets/` | Stores generated datasets using the format `dataset_{partition_type}_{timestamp}.csv`. |
| `storage/models/` | Stores trained model files, model metadata, and `default_model.json`. |
| `default_model.json` | Points the prediction service to the currently registered default model path, model type, model config, and selected threshold. |

#### `securebank/modules/data/`

This module contains model-agnostic data loading, cleaning, validation, and preprocessing logic.

| File | Purpose |
|---|---|
| `raw_data_handler.py` | Loads raw customer, transaction, and fraud files. It builds training and prediction DataFrames, merges customer information, fills missing values, drops duplicates, converts date fields, and summarizes data quality. |
| `preprocessing.py` | Orchestrates the preprocessing flow for training and prediction. It calls the raw data handler, validates schemas, fills missing values, converts dates, and writes preprocessing logs. |
| `schemas.py` | Defines required columns, default fill values, and validation checks for training and prediction input data. |

#### `securebank/modules/models/`

This module contains the model wrapper, model implementations, and model-specific adapters. The system currently supports logistic regression and gradient boosting.

| File or Directory | Purpose |
|---|---|
| `model_wrapper.py` | Provides a common interface for model training, prediction, saving, loading, and feature preparation. It routes work to the selected model implementation. |
| `logistic_regression/adapter.py` | Performs logistic regression-specific feature preparation, including categorical encoding, frequency encoding, numeric scaling, age creation, and feature alignment. |
| `logistic_regression/model.py` | Wraps the scikit-learn logistic regression model behind a consistent train/predict/save/load interface. Logistic regression is supported as the baseline model. |
| `gradient_boosting/adapter.py` | Performs gradient boosting-specific feature preparation, including categorical encoding, frequency encoding, age creation, and feature alignment. |
| `gradient_boosting/model.py` | Wraps the histogram gradient boosting classifier behind the same train/predict/save/load interface. Gradient boosting is the current default model configuration. |

#### `securebank/modules/train/`

This module owns the lower-level training workflow.

| File | Purpose |
|---|---|
| `training_orchestrator.py` | Coordinates preprocessing, dataset splitting, model-specific feature preparation, model training, and model saving. |
| `splitter.py` | Splits cleaned data into train, validation, and test sets. It supports holdout and time-based splitting strategies. |

#### `securebank/modules/predict/`

This module owns the lower-level prediction workflow.

| File | Purpose |
|---|---|
| `prediction_orchestrator.py` | Coordinates prediction preprocessing, customer enrichment, model loading, feature adaptation, and fraud prediction. |

#### `securebank/modules/services/`

This module connects Flask endpoints to the lower-level workflows. These services keep `app.py` small and make endpoint behavior easier to test.

| File | Purpose |
|---|---|
| `dataset_service.py` | Supports `/create_dataset`. It runs preprocessing, saves the cleaned dataset, returns a dataset summary, and logs the request. |
| `training_service.py` | Supports `/train_model`. It runs training, evaluates candidate thresholds, selects the best threshold, evaluates the selected threshold on test data, saves metadata, and updates `default_model.json`. |
| `prediction_service.py` | Supports `/predict`. It reads `default_model.json`, builds the prediction orchestrator, runs prediction, converts numeric labels into `legitimate` or `fraudulent`, and logs the request. |

#### `securebank/modules/evaluation/`

| File | Purpose |
|---|---|
| `evaluator.py` | Computes model evaluation metrics such as accuracy, balanced accuracy, precision, recall, F1, ROC AUC, average precision, and confusion matrix values. |

#### `securebank/modules/log/`

| File | Purpose |
|---|---|
| `logger.py` | Writes structured JSON logs to `securebank/logs/` using timestamped filenames. |


## Setup and Running Instructions

### Prerequisites

This project is intended to run through Docker. The Docker image uses `python:3.9-slim`, so users do not need to manually install Python dependencies when running the container.

Required local tools:

```text
Docker
Bash-compatible terminal
```

### 1. Add Required Data Files
Before running the system, place the SecureBank raw data files inside `securebank/data_sources/`, if not present in the directory.

```text
securebank/data_sources/customer_release.csv
securebank/data_sources/transactions_release.parquet
securebank/data_sources/fraud_release.json
```

For local development or running tests outside Docker, use Python 3.9 or a compatible Python version and install dependencies from:`pip install -r securebank/requirements.txt`

### 2. Build and Start the Docker Server
From the repository root, run:

```bash
bash securebank/executables/run_server.sh
```
This script builds the Docker image and starts the Flask API container on port 5000.

### 3. Generate a Dataset
Run the dataset generation script:

```bash
bash securebank/executables/create_dataset.sh
```

This calls POST /create_dataset, runs the preprocessing pipeline, and saves a cleaned dataset under: `securebank/storage/datasets/` in the format: `dataset_{partition_type}_{timestamp}.csv`

### 4. Train the Default Model
After generating the dataset, train and register the default model:

```bash
bash securebank/executables/train_model.sh
```
This calls `POST /train_model`, trains the configured gradient boosting model, evaluates candidate thresholds, saves the trained model and metadata, and updates:`securebank/storage/models/default_model.json`

### 5. Run a Prediction
After a default model has been trained and registered, run:

```bash
bash securebank/executables/predict.sh
```

## API Endpoints
The Flask API exposes endpoints for health checking, dataset generation, model training, and fraud prediction.

### `GET /`

Returns a basic status response confirming that the API is running.

Example request:

```bash
curl http://127.0.0.1:5000/
```

Example response:

```json
{
  "endpoints": ["/predict", "/create_dataset", "/train_model"],
  "service": "SecureBank API",
  "status": "running"
}
```

### `POST /create_dataset`

Generates a cleaned training dataset from the raw SecureBank data sources.

Example request:

```bash
curl -X POST http://127.0.0.1:5000/create_dataset \
     -H "Content-Type: application/json" \
     -d '{"partition_type": "full"}'
```

Example response fields:

```text
status
dataset_path
partition_type
created_at
dataset_summary
```

The generated dataset is saved under:

```text
securebank/storage/datasets/
```

### `POST /train_model`

Trains the configured fraud detection model, evaluates candidate thresholds, saves the trained model and metadata, and registers the model as the default model for prediction.

Example request:

```bash
curl -X POST http://127.0.0.1:5000/train_model \
     -H "Content-Type: application/json" \
     -d '{"split_strategy": "holdout", "register_as_default": true}'
```

Example response fields:

```text
status
model_path
metadata_path
default_model_updated
selected_threshold
selected_validation_result
test_result
```

The trained model and metadata are saved under:

```text
securebank/storage/models/
```

### `POST /predict`

Predicts whether a transaction is `legitimate` or `fraudulent`.

Required input fields:

```text
trans_date_trans_time
cc_num
unix_time
merchant
category
amt
merch_lat
merch_long
```

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @securebank/test.json
```

Example response:

```json
{
  "fraud_probability": 0.0017725609993923485,
  "model_type": "gradient_boosting",
  "predict": "legitimate",
  "prediction": 0,
  "threshold": 0.99
}
```

## Generated Artifacts
The system creates runtime artifacts when dataset generation, model training, and prediction workflows are executed. These artifacts support reproducibility and monitoring, but they should not be committed to Git.

### Datasets

Generated datasets are saved under:

```text
securebank/storage/datasets/
```

Dataset files use the format:

```text
dataset_{partition_type}_{timestamp}.csv
```

### Models

Trained models and model metadata are saved under:

```text
securebank/storage/models/
```

Model files are saved with versioned filenames so that each training run can be reproduced or inspected later. The `default_model.json` file stores the active model path, model type, model configuration, and selected threshold used by the `/predict` endpoint.

### Logs

Logs are saved under:

```text
securebank/logs/
```

Each log is written as a separate timestamped JSON file using the format:

```text
log_{timestamp}.json
```

Logs capture endpoint activity, request metadata, response metadata, status, and errors when applicable.

## Model Notes

The system currently supports two model workflows: logistic regression and gradient boosting. Logistic regression is available as a baseline model, while gradient boosting is configured as the default model for the deployed prediction workflow.

The default gradient boosting model uses both transaction and customer information as features. The preprocessing pipeline merges raw transaction data with customer profile data, fills missing values, validates required fields, and creates time-based features from `trans_date_trans_time`. Model-specific adapters then prepare the cleaned data for training or prediction.

During training, the system evaluates candidate fraud probability thresholds and selects the best threshold based on validation performance. The selected threshold is saved with the model metadata and reused by `/predict` through `default_model.json`.

In the latest smoke-tested training run, the default gradient boosting configuration selected a `0.99` threshold. On the validation split used for threshold selection, it achieved approximately `0.897` precision, `0.711` recall, and `0.793` F1. On the held-out test split, it produced approximately `0.865` precision, `0.666` recall, and `0.753` F1.

## Troubleshooting
Most runtime issues come from one of three places: the API container is not running, the raw data files are missing, or a default model has not been trained yet.

Start by checking the server:

```bash
docker ps
curl http://127.0.0.1:5000/
```

If the API does not respond, inspect the container logs:

```bash
docker ps -a
docker logs <container_id>
```

### Common Errors and Fixes

#### Missing Data Files

If `/create_dataset` or `/train_model` fails, confirm that the required files exist in `securebank/data_sources/`:

```text
customer_release.csv
transactions_release.parquet
fraud_release.json
```

Fix: add the missing files and rerun the script.

#### Missing Default Model

If `/predict` fails because `default_model.json` or a model file cannot be found, train the model first:

```bash
bash securebank/executables/train_model.sh
```

Then rerun prediction:

```bash
bash securebank/executables/predict.sh
```

#### Port 5000 Already in Use

If Docker cannot start because port `5000` is already in use, stop the conflicting container or process:

```bash
docker ps
docker stop <container_id>
```

Then restart the server:

```bash
bash securebank/executables/run_server.sh
```

#### Container Exits Immediately

If the container exits after starting, check the logs:

```bash
docker ps -a
docker logs <container_id>
```

Fix the reported error, then rebuild and rerun the server.