#!/bin/bash
#SBATCH --job-name=fl_manufacturing
#SBATCH --output=logs/fl_%j.out
#SBATCH --error=logs/fl_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=mit_normal

module load miniforge

source /orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/activate

mkdir -p logs

/orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/python server.py &
sleep 5
/orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/python client.py 0 &
/orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/python client.py 1 &
/orcd/home/002/kcnakamu/s26_urop/FL-for-manufacturing/.venv/bin/python client.py 2 &

wait
echo "FL job complete"