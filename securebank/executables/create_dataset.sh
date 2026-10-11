#!/bin/bash

PARTITION_TYPE=${1:-full}

curl -X POST http://127.0.0.1:5000/create_dataset \
     -H "Content-Type: application/json" \
     -d "{
           \"partition_type\": \"$PARTITION_TYPE\"
         }"