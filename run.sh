#!/bin/bash
#SBATCH --job-name=fl_manufacturing
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1

ROUNDS=${1:-10}
EPOCHS=${2:-1}

module load miniforge
source /orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/activate


# Pre-download model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

SERVER_HOST=$(hostname)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p logs/$TIMESTAMP

python server.py --rounds $ROUNDS > logs/$TIMESTAMP/server.log 2>&1 &

echo "Waiting for server to start..."
until grep -q "gRPC server running" logs/$TIMESTAMP/server.log 2>/dev/null; do
    sleep 1
done
echo "Server is ready!"

python client.py 0 $SERVER_HOST $TIMESTAMP --epochs $EPOCHS > logs/$TIMESTAMP/client_0.log 2>&1 &
python client.py 1 $SERVER_HOST $TIMESTAMP --epochs $EPOCHS > logs/$TIMESTAMP/client_1.log 2>&1 &
python client.py 2 $SERVER_HOST $TIMESTAMP --epochs $EPOCHS > logs/$TIMESTAMP/client_2.log 2>&1 &

wait
echo "FL job complete"