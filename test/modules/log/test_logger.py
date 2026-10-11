import json
from pathlib import Path

from securebank.modules.log.logger import JsonLogger


def test_constructor_creates_log_directory(tmp_path):
    log_dir = tmp_path / "logs"

    logger = JsonLogger(log_dir=log_dir)

    assert logger.log_dir == log_dir
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_write_log_creates_json_file(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    log_path = logger.write_log(
        endpoint="/predict",
        request_data={"amt": 25.0},
        response_data={"predict": "legitimate"},
        status="success",
    )

    assert log_path.exists()
    assert log_path.parent == tmp_path
    assert log_path.name.startswith("log_")
    assert log_path.suffix == ".json"


def test_write_log_stores_expected_fields(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    log_path = logger.write_log(
        endpoint="/predict",
        request_data={"amt": 25.0},
        response_data={
            "predict": "fraudulent",
            "prediction": 1,
            "fraud_probability": 0.82,
            "threshold": 0.5,
        },
        status="success",
        metadata={"model_type": "logistic_regression"},
    )

    with log_path.open("r", encoding="utf-8") as file:
        log_data = json.load(file)

    assert "timestamp" in log_data
    assert log_data["endpoint"] == "/predict"
    assert log_data["status"] == "success"

    assert log_data["request"] == {"amt": 25.0}
    assert log_data["response"] == {
        "predict": "fraudulent",
        "prediction": 1,
        "fraud_probability": 0.82,
        "threshold": 0.5,
    }
    assert log_data["error"] is None
    assert log_data["metadata"] == {"model_type": "logistic_regression"}


def test_write_log_defaults_empty_optional_fields(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    log_path = logger.write_log(
        endpoint="/train_model",
    )

    with log_path.open("r", encoding="utf-8") as file:
        log_data = json.load(file)

    assert log_data["endpoint"] == "/train_model"
    assert log_data["status"] == "success"
    assert log_data["request"] == {}
    assert log_data["response"] == {}
    assert log_data["error"] is None
    assert log_data["metadata"] == {}


def test_write_log_stores_error_message(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    log_path = logger.write_log(
        endpoint="/predict",
        request_data={"amt": None},
        status="error",
        error="Missing required field: amt",
    )

    with log_path.open("r", encoding="utf-8") as file:
        log_data = json.load(file)

    assert log_data["endpoint"] == "/predict"
    assert log_data["status"] == "error"
    assert log_data["request"] == {"amt": None}
    assert log_data["response"] == {}
    assert log_data["error"] == "Missing required field: amt"


def test_write_log_handles_non_json_native_values(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    log_path = logger.write_log(
        endpoint="/train_model",
        response_data={
            "model_path": Path("securebank/storage/models/model.joblib"),
        },
    )

    with log_path.open("r", encoding="utf-8") as file:
        log_data = json.load(file)

    assert log_data["response"]["model_path"] == "securebank/storage/models/model.joblib"


def test_timestamp_format_is_filesystem_safe(tmp_path):
    logger = JsonLogger(log_dir=tmp_path)

    timestamp = logger._timestamp()

    assert ":" not in timestamp
    assert "/" not in timestamp
    assert "\\" not in timestamp
    assert " " not in timestamp
    assert "_" in timestamp