#!/bin/bash
#SBATCH --job-name=fl_manufacturing
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Federated run. Client count and class count are parameters so the neu6
# (6-client) and the original neu3 (3-client) partitions both run from here:
#   sbatch run.sh 10 1 fedavg data/neu6_data 0 ""            # 6 clients, 6 classes
#   sbatch run.sh 10 1 fedavg data/neu_data  0 "" 3 3        # the original 3-class run
ROUNDS=${1:-10}
EPOCHS=${2:-1}
STRATEGY=${3:-fedavg}
DATA_DIR=${4:-data/neu6_data}
SEED=${5:-0}
EXP_NAME=${6:-"disruption_neu6_${STRATEGY}_seed${SEED}"}
NUM_CLIENTS=${7:-6}
NUM_CLASSES=${8:-6}

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

# Fail early rather than after the server is up: a missing client dir would
# otherwise leave the server waiting forever for a client that cannot start.
for ((i = 0; i < NUM_CLIENTS; i++)); do
    if [ ! -f "${DATA_DIR}/client_${i}/data.yaml" ]; then
        echo "ERROR: ${DATA_DIR}/client_${i}/data.yaml not found."
        echo "Generate the split first, e.g.:"
        echo "  python utils/dataset_creation/split_neu_data.py \\"
        echo "      --src data/NEU-DET-SOURCE --out ${DATA_DIR} --preset neu6"
        exit 1
    fi
done

# Pre-download model (must match MODEL_PATH in model.py)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

SERVER_HOST=$(hostname)

mkdir -p "$LOG_DIR"

# Note: #SBATCH --output/--error headers above use logs/fl_%j.out (Slurm scheduler logs).
# Application-level logs go to $LOG_DIR/ below.

echo "FL: ${NUM_CLIENTS} clients, ${NUM_CLASSES} classes, ${ROUNDS} rounds, data=${DATA_DIR}"

SHAPLEY_LOG_DIR="${FL_DIR}/shapley_logs"
python server.py --rounds $ROUNDS --strategy $STRATEGY --mu $MU --seed $SEED \
    --num_classes $NUM_CLASSES --num_clients $NUM_CLIENTS \
    --log_dir "$SHAPLEY_LOG_DIR" > "$LOG_DIR/server.log" 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" "$LOG_DIR/server.log" 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

for ((i = 0; i < NUM_CLIENTS; i++)); do
    python client.py $i $SERVER_HOST --out_dir "$FL_DIR" --epochs $EPOCHS \
        --num_classes $NUM_CLASSES --strategy $STRATEGY --data_dir $DATA_DIR \
        --seed $SEED > "$LOG_DIR/client_${i}.log" 2>&1 &
done

wait
echo "FL complete. Outputs: $EXP_DIR"
