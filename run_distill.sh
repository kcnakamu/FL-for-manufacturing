#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Arm (a): KD as a post-disruption fine-tune.
#
# The disruption condition, stated explicitly because it defines the experiment:
# client_4 (C5) departs, so its DATA is gone from the training pool
# (data/neu6s_survivors holds ZERO scratches boxes) while its FROZEN teacher
# checkpoint survives in the bank. The student starts from the pre-disruption
# federated global and keeps training on what is left. The KD term is then the
# only scratches signal in the entire objective -- which is the whole question:
# without it the class can only be forgotten.
#
# Three arms, one axis. They differ ONLY in how the KD term is weighted:
#   nokd        --lam 0        KD computed and logged but inert -> plain fine-tune
#   uniform     bank, no weights   uniform over teachers AND classes (competence-blind)
#   competence  bank + kd_weights  variance-scaled routing at tau=3
#
#   sbatch run_distill.sh nokd
#   sbatch run_distill.sh uniform
#   sbatch run_distill.sh competence
ARM=${1:?usage: sbatch run_distill.sh {nokd|uniform|competence}}
EPOCHS=${2:-75}
SEED=${3:-0}

STUDENT=experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt
DATA=data/neu6s_survivors/data.yaml
BANK=experiments/teacher_bank_neu6s_seed0/teacher_bank
KDW=experiments/competence_across_seeds_neu6s/kd_weights.json
# Must equal the bank's training imgsz: teachers are re-run on the student's
# batches, so a mismatch evaluates every teacher off its own resolution.
IMGSZ=640
OUT=experiments/kd_neu6s/${ARM}

case "$ARM" in
  nokd)       EXTRA="--lam 0.0" ;;
  uniform)    EXTRA="--lam 1.0" ;;
  competence) EXTRA="--lam 1.0 --kd_weights ${KDW}" ;;
  *) echo "unknown arm '${ARM}' (expected nokd|uniform|competence)"; exit 2 ;;
esac

set -euo pipefail

module load miniforge
source "${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}/bin/activate"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "[run_distill] arm=${ARM} epochs=${EPOCHS} seed=${SEED} -> ${OUT}"

python -m adaptation.distill_finetune \
    --weights "${STUDENT}" \
    --data "${DATA}" \
    --teacher_bank "${BANK}" \
    --out_dir "${OUT}" \
    --mode neck_head --epochs "${EPOCHS}" --lr 1e-4 \
    --imgsz "${IMGSZ}" --batch 16 --seed "${SEED}" \
    --device 0 ${EXTRA}
