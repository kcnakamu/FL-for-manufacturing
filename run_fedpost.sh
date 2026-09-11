#!/bin/bash
#SBATCH --job-name=fedpost
#SBATCH --output=logs/fedpost_%j.out
#SBATCH --error=logs/fedpost_%j.err
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No --time: mit_normal_gpu rejects an explicit wall clock, same as run.sh.

# Arm (a), FEDERATED. After client_4 (C5, the exclusive owner of Scratches)
# departs, the five survivors keep running FedAvg from the pre-departure global,
# and each distils from the frozen pre-departure teacher bank on its OWN data.
#
# This replaces the centralized version (run_distill.sh), which pooled the
# survivors' images into one fine-tune -- something a real federation cannot do.
# Everything else is held equal so the two differ only in federation: backbone
# frozen, lr0 1e-4, lam and teacher_conf as given, and a per-client budget of
# ROUNDS x EPOCHS = 75 local epochs, matching the centralized run's 75.
#
# No retained buffer here: once C5 has left there is no client left to hold its
# images, so the federated version tests the no-retention case -- the one where
# distillation is the only option.
#
#   sbatch run_fedpost.sh competence 0
ARM=${1:-}
if [ -z "$ARM" ]; then
    echo "usage: sbatch run_fedpost.sh nokd|uniform|argmax|competence [seed] [rounds] [epochs] [lam] [tconf]"
    exit 2
fi
SEED=${2:-0}
ROUNDS=${3:-15}
EPOCHS=${4:-5}
LAM=${5:-10.0}
TCONF=${6:-0.25}

INIT=experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt
BANK=experiments/teacher_bank_neu6s_seed0/teacher_bank
SURV=data/neu6s_fedsurv
case "$ARM" in
  nokd|uniform) KDW="" ;;
  argmax)       KDW=experiments/competence_across_seeds_neu6s/kd_weights_argmax.json ;;
  competence)   KDW=experiments/competence_across_seeds_neu6s/kd_weights.json ;;
  *) echo "unknown arm '${ARM}' (expected nokd|uniform|argmax|competence)"; exit 2 ;;
esac

# FEDPOST_ROOT redirects a run away from the results tree -- used for smoke
# tests, whose directory names would otherwise match the naming convention
# scripts/summarize_arms.py discovers and land in the results table.
EXP_DIR=${FEDPOST_ROOT:-experiments/kd_neu6s}/${ARM}_fedsurv_lam${LAM}_tc${TCONF}_seed${SEED}
FL_DIR=${EXP_DIR}/fl
LOG_DIR=${FL_DIR}/logs

set -euo pipefail
module load miniforge
source "${FL_VENV:-${SLURM_SUBMIT_DIR:-$(pwd)}/.venv}/bin/activate"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# The survivors' directory is built once, ahead of time (client_0..3 -> original
# 0..3, client_4 -> original 5). Check it rather than rebuild it here: concurrent
# jobs would otherwise race on the same symlinks.
NUM_CLIENTS=0
for ((i = 0; i < 6; i++)); do [ -f "${SURV}/client_${i}/data.yaml" ] && NUM_CLIENTS=$((NUM_CLIENTS + 1)); done
if [ "$NUM_CLIENTS" -ne 5 ]; then echo "ERROR: expected 5 survivor clients in ${SURV}, found ${NUM_CLIENTS}"; exit 1; fi
for f in "$INIT" "$BANK/manifest.json" ${KDW:+"$KDW"}; do
    [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
SERVER_HOST=$(hostname)
PORT=$(( 20000 + ${SLURM_JOB_ID:-$$} % 30000 ))
mkdir -p "$LOG_DIR"
echo "[fedpost] arm=${ARM} seed=${SEED} rounds=${ROUNDS} epochs=${EPOCHS} lam=${LAM} tconf=${TCONF} port=${PORT} -> ${EXP_DIR}"

# No --log_dir: Shapley update logging is for the pre-departure contribution
# analysis and costs ~42 MB a round.
python server.py --rounds "$ROUNDS" --strategy fedavg --seed "$SEED" --num_classes 6 \
    --num_clients "$NUM_CLIENTS" --port "$PORT" --init_weights "$INIT" \
    > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
WAITED=0
until grep -q "gRPC server running" "$LOG_DIR/server.log" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "ERROR: server exited early"; tail -20 "$LOG_DIR/server.log"; exit 1; fi
    if [ "$WAITED" -ge 300 ]; then echo "ERROR: server not ready after ${WAITED}s"; kill "$SERVER_PID"; exit 1; fi
    sleep 1; WAITED=$((WAITED + 1))
done
echo "Server is ready."

for ((i = 0; i < NUM_CLIENTS; i++)); do
    python client.py $i "$SERVER_HOST" --out_dir "$FL_DIR" --epochs "$EPOCHS" --num_classes 6 \
        --strategy fedavg --data_dir "$SURV" --imgsz 640 --port "$PORT" --seed "$SEED" \
        --lr0 1e-4 --freeze_mode neck_head --kd_mode "$ARM" --kd_bank "$BANK" \
        ${KDW:+--kd_weights "$KDW"} --kd_lam "$LAM" --kd_tconf "$TCONF" \
        > "$LOG_DIR/client_${i}.log" 2>&1 &
done
wait
echo "fedpost complete -> ${FL_DIR}/final_model"
