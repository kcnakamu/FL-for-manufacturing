#!/bin/bash
#SBATCH --job-name=fl_smoke
#SBATCH --output=logs/fl_smoke_%j.out
#SBATCH --error=logs/fl_smoke_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#
# Cheap sanity check for the chore/cleanup branch: 1 FL round, 1 local epoch,
# 3 clients, seed 0. Confirms the pipeline still STARTS, connects, trains one
# round, evaluates, and writes final_model/ — WITHOUT committing to a full run.
# This exercises the exact server.py / client.py CLI that run.sh uses, so if
# the cleanup broke an entry point or import, this fails fast.
#
# Submit:  sbatch smoke.sbatch
# Outputs: experiments/smoke_test/  (git-ignored)

ROUNDS=1
EPOCHS=1
STRATEGY=fedavg
DATA_DIR=${1:-data/neu_data}   # override with a smaller dir if you have one
SEED=0
EXP_NAME=smoke_test

MU=0.01

EXP_DIR="experiments/${EXP_NAME}"
FL_DIR="${EXP_DIR}/fl"
LOG_DIR="${FL_DIR}/logs"

module load miniforge
source /orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/activate

# Pre-download model (must match MODEL_PATH in model.py)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

SERVER_HOST=$(hostname)

mkdir -p "$LOG_DIR"

python server.py --rounds $ROUNDS --strategy $STRATEGY --mu $MU --seed $SEED > "$LOG_DIR/server.log" 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" "$LOG_DIR/server.log" 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

python client.py 0 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_0.log" 2>&1 &
python client.py 1 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_1.log" 2>&1 &
python client.py 2 $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS --num_classes 1 --strategy $STRATEGY --data_dir $DATA_DIR --seed $SEED > "$LOG_DIR/client_2.log" 2>&1 &

wait
echo "Smoke test complete. Check for final weights under: ${FL_DIR}/final_model/"
ls -la "${FL_DIR}/final_model/" 2>/dev/null || echo "WARNING: no final_model/ produced — inspect ${LOG_DIR}/*.log"
