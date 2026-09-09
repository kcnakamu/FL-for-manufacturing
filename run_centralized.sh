#!/bin/bash
#SBATCH --job-name=centralized
#SBATCH --output=logs/centralized_%j.out
#SBATCH --error=logs/centralized_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No --time: mit_normal_gpu rejects an explicit wall clock, same as run.sh.

# The centralized parity target: one model on the pooled union of every client's
# data. This is the number the KD claim is stated against, so it runs the SAME
# recipe as the teachers and the federated clients (model.LOCAL_TRAIN_HP, wired
# in scripts/train_centralized.py) and differs only in who holds the images.
#
# Build the pooled dataset first -- it is not the partition directory:
#   python utils/dataset_creation/pool_centralized.py \
#       --partition data/neu6s_data --out data/neu6s_centralized
#
#   sbatch run_centralized.sh data/neu6s_centralized/data.yaml \
#          experiments/baselines/centralized_neu6s 100
DATA=${1:-data/neu6s_centralized/data.yaml}
OUT=${2:-experiments/baselines/centralized_neu6s}
EPOCHS=${3:-100}
MODE=${4:-full}
LR=${5:-0.01}
SEED=${6:-0}
IMGSZ=${7:-640}

set -euo pipefail

module load miniforge
source "${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}/bin/activate"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "[run_centralized] data=${DATA} out=${OUT} epochs=${EPOCHS} mode=${MODE} lr=${LR} seed=${SEED}"

python scripts/train_centralized.py \
    --data "${DATA}" --output_dir "${OUT}" \
    --mode "${MODE}" --epochs "${EPOCHS}" --lr "${LR}" \
    --imgsz "${IMGSZ}" --batch 16 --seed "${SEED}" \
    --num_classes 6 --device 0 --workers 4
