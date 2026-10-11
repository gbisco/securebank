from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from securebank.modules.services.dataset_service import DatasetService
from securebank.modules.services.prediction_service import PredictionService
from securebank.modules.services.training_service import TrainingService

app = Flask(__name__)

SECUREBANK_ROOT = Path(__file__).resolve().parent

DATA_SOURCES_DIR = SECUREBANK_ROOT / "data_sources"
DATASET_STORAGE_DIR = SECUREBANK_ROOT / "storage" / "datasets"
MODEL_STORAGE_DIR = SECUREBANK_ROOT / "storage" / "models"
DEFAULT_MODEL_METADATA_PATH = MODEL_STORAGE_DIR / "default_model.json"

DEFAULT_MODEL_TYPE = "gradient_boosting"
DEFAULT_MODEL_CONFIG: dict[str, Any] = {}


dataset_service = DatasetService(
    data_sources_path=str(DATA_SOURCES_DIR),
    dataset_storage_path=str(DATASET_STORAGE_DIR),
)

training_service = TrainingService(
    data_sources_path=str(DATA_SOURCES_DIR),
    model_storage_path=str(MODEL_STORAGE_DIR),
    default_model_metadata_path=str(DEFAULT_MODEL_METADATA_PATH),
    model_type=DEFAULT_MODEL_TYPE,
    model_config=DEFAULT_MODEL_CONFIG,
)

prediction_service = PredictionService(
    data_sources_path=str(DATA_SOURCES_DIR),
    default_model_metadata_path=str(DEFAULT_MODEL_METADATA_PATH),
)


@app.get("/")
def index():
    return jsonify({
        "status": "running",
        "service": "SecureBank API",
        "endpoints": [
            "/predict",
            "/create_dataset",
            "/train_model",
        ],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts whether an incoming transaction is legitimate or fraudulent.
    """
    try:
        input_data = request.get_json(silent=True)

        if not input_data:
            return jsonify({
                "status": "error",
                "error": "Request body must be valid JSON.",
            }), 400

        result = prediction_service.predict_transaction(input_data)

        return jsonify(result), 200

    except FileNotFoundError as error:
        return jsonify({
            "status": "error",
            "error": str(error),
            "hint": "Run /train_model first to create default_model.json.",
        }), 500

    except Exception as error:
        return jsonify({
            "status": "error",
            "error": str(error),
        }), 500


@app.route("/create_dataset", methods=["POST"])
def create_dataset():
    """
    Creates a cleaned training dataset from available raw data sources.
    """
    try:
        request_data = request.get_json(silent=True) or {}

        result = dataset_service.create_dataset(
            customer_filename=request_data.get(
                "customer_filename",
                DatasetService.DEFAULT_CUSTOMER_FILE,
            ),
            transaction_filename=request_data.get(
                "transaction_filename",
                DatasetService.DEFAULT_TRANSACTION_FILE,
            ),
            fraud_filename=request_data.get(
                "fraud_filename",
                DatasetService.DEFAULT_FRAUD_FILE,
            ),
            partition_type=request_data.get("partition_type", "full"),
        )

        return jsonify(result), 200

    except Exception as error:
        return jsonify({
            "status": "error",
            "error": str(error),
        }), 500


@app.route("/train_model", methods=["POST"])
def train_model():
    """
    Trains a model, selects the best threshold, saves metadata,
    and updates the default model pointer.
    """
    try:
        request_data = request.get_json(silent=True) or {}

        result = training_service.train_and_select_model(
            customer_filename=request_data.get(
                "customer_filename",
                TrainingService.DEFAULT_CUSTOMER_FILE,
            ),
            transaction_filename=request_data.get(
                "transaction_filename",
                TrainingService.DEFAULT_TRANSACTION_FILE,
            ),
            fraud_filename=request_data.get(
                "fraud_filename",
                TrainingService.DEFAULT_FRAUD_FILE,
            ),
            split_strategy=request_data.get("split_strategy", "holdout"),
            thresholds=request_data.get("thresholds"),
            min_precision=float(request_data.get("min_precision", 0.70)),
            min_recall=float(request_data.get("min_recall", 0.70)),
            register_as_default=bool(
                request_data.get("register_as_default", True)
            ),
        )

        return jsonify(result), 200

    except Exception as error:
        return jsonify({
            "status": "error",
            "error": str(error),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)