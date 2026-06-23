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
EXP_NAME=${5:-"disruption_neu_${STRATEGY}"}

MU=0.01

EXP_DIR="experiments/${EXP_NAME}"
FL_DIR="${EXP_DIR}/fl"
LOG_DIR="${FL_DIR}/logs"

module load miniforge
source /orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/activate


# Pre-download model
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

SERVER_HOST=$(hostname)

mkdir -p "$LOG_DIR"

# Note: #SBATCH --output/--error headers above use logs/fl_%j.out (Slurm scheduler logs).
# Application-level logs go to $LOG_DIR/ below.

python server.py --rounds $ROUNDS --strategy $STRATEGY --mu $MU > "$LOG_DIR/server.log" 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" "$LOG_DIR/server.log" 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

python client.py 0 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > "$LOG_DIR/client_0.log" 2>&1 &
python client.py 1 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > "$LOG_DIR/client_1.log" 2>&1 &
python client.py 2 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR > "$LOG_DIR/client_2.log" 2>&1 &

wait
echo "FL complete. Outputs: $EXP_DIR"

