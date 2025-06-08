#!/bin/bash

# Check if exactly 3 parameters are provided
if [ $# -ne 3 ]; then
  echo "Error: run.sh requires exactly 3 parameters: trip_duration_days miles_traveled total_receipts_amount"
  exit 1
fi

# Assign parameters
TRIP_DURATION_DAYS=$1
MILES_TRAVELED=$2
TOTAL_RECEIPTS_AMOUNT=$3

# Validate parameters are numeric and non-negative
if ! [[ "$TRIP_DURATION_DAYS" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
   ! [[ "$MILES_TRAVELED" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
   ! [[ "$TOTAL_RECEIPTS_AMOUNT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Error: All parameters must be numeric and non-negative"
  exit 1
fi

# Call Python script to make prediction
# Assumes models/ directory is in the same directory as run.sh
PREDICTION=$(python predict_single.py "$TRIP_DURATION_DAYS" "$MILES_TRAVELED" "$TOTAL_RECEIPTS_AMOUNT")

# Check if Python script executed successfully
if [ $? -ne 0 ]; then
  echo "Error: Prediction failed"
  exit 1
fi

# Output the prediction formatted to 4 decimal places
printf "%.4f\n" "$PREDICTION"