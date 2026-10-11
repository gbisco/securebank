#!/bin/bash

MODEL_TYPE=${1:-gradient_boosting}
SPLIT_STRATEGY=${2:-holdout}

curl -X POST http://127.0.0.1:5000/train_model \
     -H "Content-Type: application/json" \
     -d "{
           \"model_type\": \"$MODEL_TYPE\",
           \"split_strategy\": \"$SPLIT_STRATEGY\",
           \"register_as_default\": true
         }"