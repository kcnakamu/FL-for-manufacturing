# Federated Learning for Manufacturing Defect Detection

This project trains a YOLOv8 object detector with Federated Learning (Flower) across three simulated manufacturing factories. The primary experiment evaluates **disruption-aware FL**: after 10 rounds of collaborative training, the two main factories go offline and the low-data backup factory adapts using the shared global model.

---

## Project Structure

```
.
├── client.py                  # FL client: local YOLO training + evaluation per round
├── server.py                  # FL server: aggregation strategies (FedAvg, FedProx, etc.)
├── model.py                   # Load YOLOv8n, get/set parameters
├── data.py                    # Return dataset YAML path for a given client folder
├── run.sh                     # SLURM script: launch server + 3 clients
├── requirements.txt
│
├── scripts/
│   ├── train_centralized.py   # Staged fine-tuning (head-only, neck+head, full)
│   ├── analyze_results.py     # Evaluate FL + adaptation checkpoints, write results.json
│   └── analysis/              # Existing results and visualizations
│       ├── results.json
│       ├── figures/
│       └── visualize_results.ipynb
│
├── utils/
│   ├── analysis/
│   │   ├── tune_threshold.py  # Sweep YOLO confidence threshold, find optimal value
│   │   ├── evaluate_test.py   # Evaluate a single checkpoint on a test split
│   │   ├── plot_metrics.py    # Plot FL round metrics from logs
│   │   └── dataset_summary.py # Print class/split counts for a dataset
│   ├── data/
│   │   ├── centeralized_dataset.py  # Merge client folders into one centralized dataset
│   │   └── update_yaml_paths.py     # Fix absolute paths in data.yaml files
│   └── dataset_creation/
│       ├── split_neu_data.py  # Split NEU-DET into federated client folders
│       └── coating_vision.py  # Dataset creation for coating defect data
│
├── notebooks/
│   ├── visualize_neu_classes.ipynb  # Visualize NEU-DET defect classes with bounding boxes
│   └── neu_data_validation.ipynb
│
└── experiments/               # All experiment outputs (git-ignored)
    └── <exp_name>/
        ├── fl/                # FL rounds 1–10: round_*/client_*/, final_model/, logs/
        ├── adaptation/        # Client 2 staged fine-tuning: head_only/, neck_head/, full/
        ├── baselines/         # Comparison models: centralized/, client_only/, fl_full/
        └── analysis/          # results.json, threshold_sweep.json, figures/
```

---

## Dataset

**NEU Surface Defect Database** — 6 classes of hot-rolled steel strip defects, 300 images per class (240 train / 60 val). Three classes are used in this experiment:

| Class | Description |
|-------|-------------|
| Inclusion | Embedded foreign particles — visually distinct blobs |
| Patches | Area discoloration / surface patches |
| Scratches | Linear surface scratches |

**Data split across 3 simulated factories (clients):**

| | Inclusion | Patches | Scratches | Total | Share |
|-|-----------|---------|-----------|-------|-------|
| Client 0 (Factory A) | 201 | 143 | 112 | 456 | 59.7% |
| Client 1 (Factory B) | 46  | 104 | 78  | 228 | 29.8% |
| Client 2 (Factory C) | 7   | 8   | 65  | 80  | 10.4% |
| **Total train**      | 254 | 255 | 255 | 764 | |

Central test set: 45 images (15 per class, balanced). Each client val: 30 images (10 per class).

**Generate the dataset split:**
```bash
python utils/dataset_creation/split_neu_data.py \
    --src NEU-DET \
    --out data/neu_data
```

---

## Experiment: Disruption-Aware FL

### Stage 0 — Pre-disruption FL (Rounds 1–10)
All three clients train collaboratively with FedAvg.

```bash
# SLURM
sbatch run.sh 10 5 fedavg data/neu_data disruption_neu_fedavg

# Local (three separate terminals)
python server.py --rounds 10 --strategy fedavg
python client.py 0 localhost --out_dir experiments/disruption_neu_fedavg/fl --epochs 5 --data_dir data/neu_data
python client.py 1 localhost --out_dir experiments/disruption_neu_fedavg/fl --epochs 5 --data_dir data/neu_data
python client.py 2 localhost --out_dir experiments/disruption_neu_fedavg/fl --epochs 5 --data_dir data/neu_data
```

`run.sh` arguments: `ROUNDS EPOCHS STRATEGY DATA_DIR EXP_NAME`

### Stage 1 — Head-Only Adaptation (Rounds 11–15)
Client 2 initializes from the Round-10 global model. Backbone frozen, only detection head trained.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
    --mode    head_only --epochs 25 --lr 0.001 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

### Stage 2 — Neck + Head Fine-Tuning (Rounds 16–30)
Backbone still frozen; neck and head unfrozen.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/adaptation/head_only/weights/best.pt \
    --mode    neck_head --epochs 75 --lr 0.0001 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

### Stage 3 — Full Fine-Tuning (optional)
Only run if Stage 2 still underperforms. Risk of overfitting on Client 2's small dataset.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/adaptation/neck_head/weights/best.pt \
    --mode    full --epochs 20 --lr 0.00001 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

---

## Threshold Tuning

After training, find the optimal YOLO confidence threshold on the test set:

```bash
python utils/analysis/tune_threshold.py \
    --model  experiments/disruption_neu_fedavg/adaptation/neck_head/weights/best.pt \
    --data   data/neu_data/test/data.yaml \
    --split  test \
    --metric f1 \
    --output experiments/disruption_neu_fedavg/analysis/threshold_sweep.json
```

Sweeps `conf` from 0.10 to 0.90 (step 0.05). Writes per-threshold metrics + best threshold to JSON.

---

## Analyze Results

Evaluate all checkpoints (Round 10 global, Stage 1, Stage 2) against Client 2 val and the central test set:

```bash
python scripts/analyze_results.py \
    --round10      experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
    --stage1       experiments/disruption_neu_fedavg/adaptation/head_only/weights/best.pt \
    --stage2       experiments/disruption_neu_fedavg/adaptation/neck_head/weights/best.pt \
    --client2_data data/neu_data/client_2/data.yaml \
    --central_data data/neu_data/test/data.yaml \
    --output       experiments/disruption_neu_fedavg/analysis/results.json
```

---

## Baselines

Reuses `scripts/train_centralized.py` with different `--output_dir` targets.

**Centralized model** — all client data pooled, standard YOLO training:
```bash
python scripts/train_centralized.py \
    --data data/neu_data/all_clients/data.yaml \
    --mode full --epochs 150 --lr 0.01 \
    --output_dir experiments/disruption_neu_fedavg/baselines/centralized
```

**Client-only models** — each factory trains independently, no federation:
```bash
for i in 0 1 2; do
  python scripts/train_centralized.py \
      --data data/neu_data/client_${i}/data.yaml \
      --mode full --epochs 150 --lr 0.01 \
      --output_dir experiments/disruption_neu_fedavg/baselines/client_only/client_${i}
done
```

**Full FL baseline** — standard 30-round FL, no disruption:
```bash
sbatch run.sh 30 5 fedavg data/neu_data fl_full_baseline
```

---

## Requirements

- Python 3.10+
- CUDA GPU recommended
- See `requirements.txt`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first use.
