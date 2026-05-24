#!/bin/bash
#SBATCH --job-name=fl_manufacturing
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1

ROUNDS=${1:-10}
EPOCHS=${2:-1}
STRATEGY=${3:-adaptive}
MU=${4:-0.01}
DATA_DIR=${5:-data}
# Optional flags passed straight to prepare_dataset.py, e.g. "--negatives --augment"
PREPARE_FLAGS=${6:-}


module load miniforge
source /orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/activate

# Prepare dataset and generate data.yaml files (skips if DATA_DIR already exists)
echo "Preparing dataset in $DATA_DIR..."
python utils/prepare_dataset.py --output_dir $DATA_DIR $PREPARE_FLAGS

# Pre-download model
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

SERVER_HOST=$(hostname)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RUN_NAME="${TIMESTAMP}_${STRATEGY}"

mkdir -p logs/$RUN_NAME

python server.py --rounds $ROUNDS --strategy $STRATEGY --mu $MU > logs/$RUN_NAME/server.log 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" logs/$RUN_NAME/server.log 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

python client.py 0 $SERVER_HOST $RUN_NAME --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > logs/$RUN_NAME/client_0.log 2>&1 &
python client.py 1 $SERVER_HOST $RUN_NAME --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > logs/$RUN_NAME/client_1.log 2>&1 &
python client.py 2 $SERVER_HOST $RUN_NAME --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > logs/$RUN_NAME/client_2.log 2>&1 &

wait
echo "FL job complete"

# Evaluate the final global model (client_0's copy) on the test set
FINAL_MODEL="fl_runs/$RUN_NAME/final_model/client_0_final.pt"
if [ -f "$FINAL_MODEL" ]; then
    echo "Running test-set evaluation..."
    python utils/evaluate_test.py \
        --model "$FINAL_MODEL" \
        --data_dir "$DATA_DIR" \
        --output_dir "fl_runs/$RUN_NAME/test_results" \
        --save_csv "logs/$RUN_NAME/test_results.csv"
else
    echo "WARNING: final model not found at $FINAL_MODEL — skipping test evaluation"
fi