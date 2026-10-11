#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_FILE="$SCRIPT_DIR/../test.json"

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @"$JSON_FILE"