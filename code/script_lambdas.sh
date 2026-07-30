#!/bin/bash

echo "Launching all 10 recommended MSE + DECORR experiments..."

# Each entry: "lambda_MSE lambda_DECORR"
EXPERIMENTS=(
    "1.0 0.01"
    "1.0 0.05"
    "1.0 0.1"
    "0.1 0.1"
    "0.1 1.0"
    "0.05 5.0"
    "0.05 9.0"
    "0.01 2.0"
    "0.01 5.0"
    "0.01 10.0"
)

for exp in "${EXPERIMENTS[@]}"; do
    set -- $exp

    LAMBDA_MSE=$1
    LAMBDA_DECORR=$2

    # Build experiment name

    echo "Running: $EXP_NAME"
    echo "  lambda_MSE: $LAMBDA_MSE"
    echo "  lambda_DECORR: $LAMBDA_DECORR"

    python3 cnn_osb.py \
        --gated \
        --lambda_MSE "$LAMBDA_MSE" \
        --lambda_DECORR "$LAMBDA_DECORR" \

    echo "--------------------------------------------------------"
done

echo "All experiments completed."
