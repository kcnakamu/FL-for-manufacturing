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
# teacher_conf is NOT optional in practice. Measured on real batches
# (scripts/verify_kd_loss.py), the unmasked term averages over ~8400 anchors
# where student and teacher already agree on background, so at lam=1 it is 1.84%
# of the loss but only 0.03% of the GRADIENT -- the first ablation ran that way
# and its three arms were mechanically the same run. Gating at 0.25 lifts the
# gradient share to 5.5% at the same lam, and lam=10 with 0.25 reaches 36.8%.
#
# Note the mask is taken on the FUSED teacher probability, so the weighting
# decides which anchors survive it: an anchor where only C5 fires at 0.9 fuses
# to ~0.15 under uniform (dropped at a 0.25 gate) but ~0.9 under competence
# routing. That is a real effect and a confound -- the arms then distill on
# different anchor sets -- and it must be reported, not discovered later.
#
#   sbatch run_distill.sh nokd       75 0 10 0.25
#   sbatch run_distill.sh uniform    75 0 10 0.25
#   sbatch run_distill.sh competence 75 0 10 0.25
# Not ${1:?...}: a "}" inside that message closes the expansion early and the
# remainder lands in the variable, which is how this first ran with ARM set
# to "competence}".
ARM=${1:-}
if [ -z "$ARM" ]; then
    echo "usage: sbatch run_distill.sh nokd|uniform|argmax|competence [epochs] [seed]"
    exit 2
fi
EPOCHS=${2:-75}
SEED=${3:-0}
LAM=${4:-10.0}
TCONF=${5:-0.25}
# The transfer set. Distillation moves only what the teacher demonstrates on the
# images it is shown, so this is a first-class variable, not a path detail:
# data/neu6s_survivors holds none of the departed class, the _buf variants keep
# back a labelled memory buffer of 10 or 25 of its images.
DATA_DIR=${6:-data/neu6s_survivors}

STUDENT=experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt
DATA=${DATA_DIR}/data.yaml
TAG=$(basename "${DATA_DIR}" | sed 's/^neu6s_//')
BANK=experiments/teacher_bank_neu6s_seed0/teacher_bank
KDW=experiments/competence_across_seeds_neu6s/kd_weights.json
# tau -> 0 endpoint of the same estimator: each class routed to its single best
# teacher. On the departed class this is IDENTICAL to competence routing (C5 at
# w=1.0, same lambda_c); the two differ only on the five shared classes. So it is
# the baseline that isolates what variance scaling adds -- uniform cannot, since
# averaging one expert with five non-experts dilutes the class by construction.
KDW_ARGMAX=experiments/competence_across_seeds_neu6s/kd_weights_argmax.json
# Must equal the bank's training imgsz: teachers are re-run on the student's
# batches, so a mismatch evaluates every teacher off its own resolution.
IMGSZ=640
# Settings are in the path so runs at different lam/teacher_conf/seed cannot
# silently overwrite each other -- the first ablation wrote to a bare arm name.
OUT=experiments/kd_neu6s/${ARM}_${TAG}_lam${LAM}_tc${TCONF}_seed${SEED}

case "$ARM" in
  # lam=0 makes KD inert; teacher_conf is then irrelevant but stays passed so
  # every arm walks an identical code path.
  nokd)       EXTRA="--lam 0.0" ;;
  uniform)    EXTRA="--lam ${LAM}" ;;
  argmax)     EXTRA="--lam ${LAM} --kd_weights ${KDW_ARGMAX}" ;;
  competence) EXTRA="--lam ${LAM} --kd_weights ${KDW}" ;;
  *) echo "unknown arm '${ARM}' (expected nokd|uniform|argmax|competence)"; exit 2 ;;
esac

set -euo pipefail

module load miniforge
source "${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}/bin/activate"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "[run_distill] arm=${ARM} data=${DATA_DIR} epochs=${EPOCHS} seed=${SEED} lam=${LAM} teacher_conf=${TCONF} -> ${OUT}"

python -m adaptation.distill_finetune \
    --weights "${STUDENT}" \
    --data "${DATA}" \
    --teacher_bank "${BANK}" \
    --out_dir "${OUT}" \
    --mode neck_head --epochs "${EPOCHS}" --lr 1e-4 \
    --imgsz "${IMGSZ}" --batch 16 --seed "${SEED}" \
    --teacher_conf "${TCONF}" \
    --device 0 ${EXTRA}
