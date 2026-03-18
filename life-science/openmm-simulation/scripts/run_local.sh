#!/bin/bash

# Simple example script to run OpenMM simulation locally (for testing)

set -e

PROTEIN_ID=${1:-"1UBQ"}
STEPS=${2:-"100"}

echo "Running OpenMM simulation for protein $PROTEIN_ID with $STEPS steps"


# Check if required environment variables are set
echo "Checking environment variables..."
required_vars=(
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "AWS_DEFAULT_REGION"
    "S3_BUCKET"
    "S3_PREFIX"
    "S3_ENDPOINT_URL"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
        echo "❌ $var is not set"
    else
        echo "✅ $var is set"
    fi
done

# Run local simulation without mutating the environment.
# setup.sh should be used once to create/install .venv.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
    python -m sim.run "$PROTEIN_ID" "$STEPS"
elif [ -x ".venv/bin/python" ]; then
    echo "Using project virtual environment: .venv"
    .venv/bin/python -m sim.run "$PROTEIN_ID" "$STEPS"
else
    echo "❌ No Python environment found for this example." >&2
    echo "Run ./scripts/setup.sh first, then retry ./scripts/run_local.sh." >&2
    exit 1
fi

echo "Simulation completed!"
