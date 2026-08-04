#!/bin/bash
#SBATCH --job-name=fl_manufacturing
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

ROUNDS=${1:-10}
EPOCHS=${2:-1}
STRATEGY=${3:-fedavg}
DATA_DIR=${4:-data}
SEED=${5:-0}
EXP_NAME=${6:-"disruption_neu_${STRATEGY}_seed${SEED}"}

MU=0.01

EXP_DIR="experiments/${EXP_NAME}"
FL_DIR="${EXP_DIR}/fl"
LOG_DIR="${FL_DIR}/logs"

module load miniforge
# Activate the project venv. Defaults to the .venv in the directory you ran
# `sbatch` from (normally the repo root); override with FL_VENV to point
# elsewhere:  FL_VENV=/path/to/.venv sbatch run.sh ...
VENV="${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}"
source "$VENV/bin/activate"


# Pre-download model (must match MODEL_PATH in model.py)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

SERVER_HOST=$(hostname)

mkdir -p "$LOG_DIR"

# Note: #SBATCH --output/--error headers above use logs/fl_%j.out (Slurm scheduler logs).
# Application-level logs go to $LOG_DIR/ below.

SHAPLEY_LOG_DIR="${FL_DIR}/shapley_logs"
python server.py --rounds $ROUNDS --strategy $STRATEGY --mu $MU --seed $SEED --log_dir "$SHAPLEY_LOG_DIR" > "$LOG_DIR/server.log" 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" "$LOG_DIR/server.log" 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

python client.py 0 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 3 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_0.log" 2>&1 &
python client.py 1 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 3 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_1.log" 2>&1 &
python client.py 2 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 3 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_2.log" 2>&1 &

wait
echo "FL complete. Outputs: $EXP_DIR"

