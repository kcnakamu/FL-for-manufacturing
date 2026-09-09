#!/bin/bash
#SBATCH --job-name=teacher_bank
#SBATCH --output=logs/teachers_%j.out
#SBATCH --error=logs/teachers_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No --time: mit_normal_gpu rejects an explicit wall clock here, same as run.sh.
# A full 6-teacher bank is ~10 min of training, so the partition default is ample.

# Train one teacher per client, then score the bank into a competence matrix.
# Partition, output root and seed are parameters so the neu6 (pitted_surface
# monopoly) and neu6s (scratches monopoly) partitions both run from here, and so
# the >=2 seeds the variance-scaled KD weights require can be produced without
# editing this file:
#
#   sbatch run_teachers.sh data/neu6_data  experiments/teacher_bank            0
#   sbatch run_teachers.sh data/neu6s_data experiments/teacher_bank_neu6s_seed0 0
#   sbatch run_teachers.sh data/neu6s_data experiments/teacher_bank_neu6s_seed1 1
#
# Each seed writes its own OUT_DIR. Aggregate them afterwards with:
#   python scripts/aggregate_competence.py <out>/competence/competence_matrix.json ...
DATA_DIR=${1:-data/neu6_data}
OUT_DIR=${2:-experiments/teacher_bank}
SEED=${3:-0}
EPOCHS=${4:-100}
IMGSZ=${5:-640}

# The competence matrix must be measured on the SAME partition the teachers were
# trained on -- it cross-checks every score against that partition's per-class
# training-box counts, so a mismatch here is caught rather than silently scored.
VAL_YAML="${DATA_DIR}/val/data.yaml"

set -euo pipefail

module load miniforge
source "${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}/bin/activate"
# Warm the COCO checkpoint before training: teachers refuse to start if their
# first backbone conv does not match yolov8n.pt, and a cold download inside the
# training loop races across concurrent jobs.
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "[run_teachers] data=${DATA_DIR} out=${OUT_DIR} seed=${SEED} epochs=${EPOCHS} imgsz=${IMGSZ}"

python scripts/train_local_teachers.py \
    --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" --imgsz "${IMGSZ}" --seed "${SEED}" \
    --device 0 --workers 4 --skip_existing

python scripts/competence_matrix.py \
    --bank "${OUT_DIR}/teacher_bank" \
    --val_yaml "${VAL_YAML}" \
    --out_dir "${OUT_DIR}/competence" \
    --data_dir "${DATA_DIR}" \
    --seed "${SEED}" \
    --imgsz "${IMGSZ}" \
    --device 0
