# Project Context — Disruption-Aware Federated Learning for Manufacturing Defect Detection

Drop this file into a new session to give full context on the project, experiment design, data, and codebase.

---

## What This Project Is

A research experiment (MIT S26 UROP) studying whether **federated learning helps a low-data backup factory** recover detection capability after the main factories go offline. The detection task is surface defect detection on steel strips using YOLOv8.

---

## Experiment Design

### The Setup
- 3 simulated factories (clients), each with their own local dataset partition
- No data is shared between clients — only model weights are communicated
- Framework: Flower (flwr) + Ultralytics YOLOv8

### The Disruption Scenario
1. **Rounds 1–10**: All 3 clients train together using FedAvg → produces a shared global model
2. **Disruption**: Client 0 and Client 1 go offline (simulating factory shutdowns)
3. **Client 2 adapts**: Initializes from the Round-10 global model and fine-tunes in stages using only its own small local dataset

### Why This Is Interesting
Client 2 has only 80 training images (10.4% of total data) and was heavily skewed toward Scratches. Alone, it performs poorly on Inclusion and Patches. The hypothesis is that the FL global model carries useful shared representations that let Client 2 adapt faster and better than training from scratch.

---

## Dataset

**Source**: NEU Surface Defect Database (Kaggle: kaustubhdikshit/neu-surface-defect-database)

**Original**: 6 defect classes × 300 images = 1,800 total (240 train / 60 val per class)

**Classes chosen** (3 of 6, selected for visual clarity):
- `Inclusion` — embedded foreign particles, visually distinct blobs
- `Patches` — area discoloration / surface patches
- `Scratches` — linear surface scratches

Classes not used: Crazing (hard to see with naked eye), Pitted Surface, Rolled-in Scale.

**Split into 3 clients + central test:**

| Split | Inclusion | Patches | Scratches | Total |
|-------|-----------|---------|-----------|-------|
| Client 0 train | 201 | 143 | 112 | 456 (59.7%) |
| Client 1 train | 46  | 104 | 78  | 228 (29.8%) |
| Client 2 train | 7   | 8   | 65  | 80  (10.4%) |
| Each client val | 10 | 10  | 10  | 30 (from val folder) |
| Central test    | 15 | 15  | 15  | 45 (from val folder) |

Client 2 is intentionally data-scarce and class-skewed (81% Scratches) to simulate a backup factory just ramping up.

**On-disk format** (`data/neu_data/`): Ultralytics YOLO-compatible — `images/{train,val}/` + `labels/{train,val}/` + `data.yaml` per client. Class IDs: 0=Inclusion, 1=Patches, 2=Scratches.

**Generate the split:**
```bash
python utils/dataset_creation/split_neu_data.py --src NEU-DET --out data/neu_data
```

---

## Adaptation Stages (Post-Disruption)

Client 2 fine-tunes from the Round-10 global model in stages to preserve the shared backbone:

| Stage | Layers Trained | Script Mode | LR | Epochs |
|-------|---------------|-------------|-----|--------|
| 1 — Head only | Detection head only | `head_only` | 0.001 | 25 |
| 2 — Neck + Head | YOLO neck + head | `neck_head` | 0.0001 | 75 |
| 3 — Full (optional) | All layers | `full` | 0.00001 | ~20 |

Stage 3 is only run if Stage 2 still underperforms — Client 2's dataset is small enough that full fine-tuning risks overfitting.

---

## Baselines

| Baseline | Description |
|----------|-------------|
| Centralized | All 764 training images pooled, standard YOLO training (no federation) |
| Client-only | Each factory trains its own model from scratch, no communication |
| FL Full (30 rounds) | Standard FL for all 30 rounds, no disruption event |

The disruption experiment is compared against all three.

---

## Evaluation Metrics

Reported at each checkpoint (Round 10 global, Stage 1, Stage 2, optionally Stage 3):
- mAP50
- Precision, Recall, F1
- False positives (total + per image)
- Per-class confusion (TP, FP, FN for each of Inclusion / Patches / Scratches)

Evaluated on two sets:
- **Client 2 validation** (30 images) — how well Client 2 generalizes on its own local distribution
- **Central test set** (45 images, balanced) — overall defect detection quality

**Threshold tuning**: The confidence threshold is swept from 0.10–0.90 to find the value maximizing F1. This is done per checkpoint since the optimal threshold can shift after fine-tuning.

---

## Codebase Structure

```
client.py                          # FL client — trains YOLO locally each round, saves metrics.json
server.py                          # FL server — FedAvg / FedProx / adaptive aggregation
model.py                           # load_model(), get_parameters(), set_parameters()
data.py                            # get_dataset_yaml(factory_dir) — returns path to data.yaml
run.sh                             # SLURM launcher: args = ROUNDS EPOCHS STRATEGY DATA_DIR EXP_NAME

scripts/
  train_centralized.py             # Staged fine-tuning for post-disruption adaptation + baselines
  analyze_results.py               # Batch evaluation of Round10 / Stage1 / Stage2 checkpoints

utils/
  analysis/
    tune_threshold.py              # Sweep YOLO conf threshold; find optimal by F1/mAP50/etc.
    evaluate_test.py               # Single-checkpoint evaluation on a test split
    plot_metrics.py                # Plot per-round FL metrics from metrics.json files
    dataset_summary.py             # Print image/class counts for a dataset
  data/
    centeralized_dataset.py        # Merge client folders into one centralized dataset folder
    update_yaml_paths.py           # Fix absolute paths in data.yaml after moving data
  dataset_creation/
    split_neu_data.py              # Split NEU-DET raw data into federated client folders
    coating_vision.py              # Dataset creation for a separate coating defect dataset

notebooks/
  visualize_neu_classes.ipynb      # Show bounding-box samples for all 6 NEU classes side-by-side
  neu_data_validation.ipynb        # Data validation notebook
```

---

## Output Structure

All experiment outputs go under `experiments/` (git-ignored):

```
experiments/<exp_name>/
  fl/
    round_00/ … round_09/
      client_{0,1,2}/              # YOLO train outputs (weights, results.csv, etc.)
      client_{n}_val/metrics.json  # Per-round eval: mAP50, P, R, F1, inference_ms
    final_model/
      client_{0,1,2}_final.pt      # Global model weights after Round 10
    logs/
      server.log, client_{0,1,2}.log
  adaptation/
    head_only/weights/{best,last}.pt
    neck_head/weights/{best,last}.pt
    full/weights/{best,last}.pt    # optional
  baselines/
    centralized/weights/{best,last}.pt
    client_only/client_{0,1,2}/weights/{best,last}.pt
    fl_full/                       # same structure as fl/ above
  analysis/
    results.json                   # Output of analyze_results.py
    threshold_sweep.json           # Output of tune_threshold.py
    figures/
```

---

## Key Design Decisions

- **Why FedAvg not FedProx for the disruption experiment?** Earlier baseline runs showed FedAvg gives higher precision at the cost of more false positives; FedProx helps but the disruption experiment isolates the adaptation effect — kept simple for now.
- **Why freeze backbone in adaptation stages?** Client 2 has only 80 images. Full fine-tuning from Round 10 risks forgetting the shared visual features learned from Clients 0 and 1.
- **Why 3 classes?** Crazing was visually ambiguous even to human inspection. Inclusion, Patches, Scratches were the most visually distinct from the 6 available.
- **Why Client 2 has so few images?** By design — it simulates a backup factory that was not fully operational. The 60/30/10 split approximates realistic data availability skew.
- **Class ID mapping**: Inclusion=0, Patches=1, Scratches=2 (set in `utils/dataset_creation/split_neu_data.py`).

---

## Existing Results (scripts/analysis/results.json)

Three-stage evaluation on the current NEU dataset (Crazing/Pitted/Rolled-in Scale — **old classes**, before the switch to Inclusion/Patches/Scratches):

| Stage | Dataset | mAP50 | Precision | Recall | F1 |
|-------|---------|-------|-----------|--------|-----|
| Round 10 global | Client 2 val | 0.344 | 0.689 | 0.220 | 0.333 |
| Round 10 global | Central test | 0.356 | 0.873 | 0.265 | 0.407 |
| Stage 1 head-only | Client 2 val | 0.138 | 0.106 | 0.420 | 0.170 |
| Stage 1 head-only | Central test | 0.239 | 0.157 | 0.372 | 0.221 |
| Stage 2 neck+head | Client 2 val | 0.206 | 0.144 | 0.340 | 0.202 |
| Stage 2 neck+head | Central test | 0.205 | 0.156 | 0.278 | 0.200 |

Note: Crazing had 0 TP across all stages — it was nearly invisible, which motivated switching to Inclusion/Patches/Scratches.

---

## What's Next

1. Re-run `split_neu_data.py` to generate the new dataset (Inclusion / Patches / Scratches)
2. Run the disruption FL experiment: `sbatch run.sh 10 5 fedavg data/neu_data disruption_neu_fedavg`
3. Run adaptation stages 1 and 2 with `scripts/train_centralized.py`
4. Tune threshold with `utils/analysis/tune_threshold.py`
5. Analyze with `scripts/analyze_results.py`
6. Run baselines for comparison
