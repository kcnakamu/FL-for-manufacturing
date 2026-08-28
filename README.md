# Federated Learning for Manufacturing Defect Detection

This project trains a YOLOv8 object detector with Federated Learning (Flower) across three simulated manufacturing factories. The primary experiment evaluates **disruption-aware FL**: after 10 rounds of collaborative training, the two main factories go offline and the low-data backup factory adapts using the shared global model.

---

## Project Structure

```
.
├── client.py                  # FL client: local YOLO training + evaluation per round
├── server.py                  # FL server: aggregation strategies (FedAvg, FedProx, etc.)
├── model.py                   # Load YOLOv8n, get/set parameters, apply_freeze()
├── data.py                    # Return dataset YAML path for a given client folder
├── run.sh                     # SLURM script: launch server + 3 clients
├── requirements.txt
│
├── strategies/                # Pluggable server aggregation (fedavg, fedprox, fedawa, adaptive)
│
├── shapley/                   # Shapley contribution-persistence analysis (see docs/)
│   ├── logger.py              # Non-invasive per-round logging of client updates + globals
│   ├── reconstruct.py         # Rebuild any coalition's model by re-aggregating logged updates
│   ├── shapley.py             # Exact N=3 Shapley values from the 8 coalition utilities
│   ├── evaluate.py            # v(S) = mAP50 of a model on the shared test set
│   ├── persistence.py         # Driver: retention curve rho_i(tau) + per-class forgetting
│   ├── convergence.py         # Disruption-timing analysis: convergence round + t* choice
│   ├── contribution.py        # Per-class contribution matrix + KD class weights
│   └── tests/                 # Unit + integration tests (run without pytest)
│
├── adaptation/                # Post-disruption adaptation of Client 2 (C)
│   ├── controller.py          # StageController: escalate freeze regime on val plateau
│   ├── adaptive_finetune.py   # Driver: adaptive head_only -> neck_head fine-tuning
│   ├── kd.py                  # Class-retention-weighted output-level knowledge distillation
│   ├── distill_finetune.py    # Driver: fine-tune C with the KD loss attached
│   └── tests/                 # Unit tests incl. Ultralytics API tripwire
│
├── docs/
│   └── shapley_explained.md   # Full walkthrough of the contribution-persistence method
│
├── scripts/
│   ├── train_centralized.py   # Staged fine-tuning (head-only, neck+head, full)
│   ├── analyze_results.py     # Evaluate FL + adaptation checkpoints, write results.json
│   └── analysis/
│       └── visualize_results.ipynb
│
├── utils/
│   ├── analysis/
│   │   ├── tune_threshold.py  # Sweep YOLO confidence threshold, find optimal value
│   │   ├── evaluate_test.py   # Evaluate checkpoint(s) on the test split
│   │   ├── plot_metrics.py    # Plot per-round mAP50 across FL rounds
│   │   ├── plot_fl_rounds.py  # Plot all per-round metrics (mAP50, P, R, F1, ...)
│   │   └── dataset_summary.py # Print class/split counts for a dataset
│   ├── data/
│   │   ├── centralized_dataset.py   # Merge client folders into one centralized dataset
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
    ├── convergence_analysis/  # Disruption-timing study (cross-experiment)
    └── <exp_name>/
        ├── fl/                # FL rounds 1–10: round_*/client_*/, final_model/, logs/
        │   └── shapley_logs/  # Per-round client updates + global checkpoints (for Shapley)
        ├── adaptation/        # Client 2 staged fine-tuning: head_only/, neck_head/, full/
        ├── adaptive/          # Adaptive stage selection: seg*/ runs + trace.json
        ├── distill/           # KD fine-tuning runs + distill_config.json
        ├── baselines/         # Comparison models: centralized/, client_only/, fl_full/
        ├── analysis/          # results.json, threshold_sweep.json, figures/
        ├── shapley/           # retention_curve.png, shapley_by_checkpoint.csv, records.json
        └── contribution/      # contribution_matrix.*, class_retention_weights.json, heatmap
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
| Client 0 (Factory A) | 202 | 143 | 112 | 457 | 59.7% |
| Client 1 (Factory B) | 46  | 104 | 78  | 228 | 29.8% |
| Client 2 (Factory C) | 7   | 8   | 65  | 80  | 10.5% |
| **Total train**      | 255 | 255 | 255 | 765 | |

Central test set: 45 images (15 per class, balanced). Each client val: 30 images (10 per class).

**Generate the dataset split:**
```bash
python utils/dataset_creation/split_neu_data.py \
    --src data/NEU-DET-SOURCE \
    --out data/neu_data \
    --preset neu3 \
    --seed 42
```

`--seed` controls *which* images land in each client/test split (the per-client
class counts are fixed by the preset). Vary it to re-shuffle the split for a
robustness check; it defaults to `42` (the original split).

### 6-class / 6-client split (`--preset neu6`)

A second partition covering **all six** NEU-DET classes across six clients, used
for the contribution/exclusive-owner experiments. Holdout is taken first —
45 test + 30 val per class, both class-balanced and centralized — leaving
**225 images per class** to distribute (224 for scratches; see below).

| | crazing | inclusion | patches | pitted | rolled-in | scratches | Total |
|-|---------|-----------|---------|--------|-----------|-----------|-------|
| `client_0` — C1 generalist          | 50  | 60  | 90  | 0   | 50  | 85 | 335 |
| `client_1` — C2 generalist          | 40  | 110 | 110 | 0   | 0   | 84 | 344 |
| `client_2` — C3 partial specialist  | 0   | 0   | 0   | 0   | 175 | 30 | 205 |
| `client_3` — C4 partial specialist  | 110 | 30  | 0   | 0   | 0   | 0  | 140 |
| `client_4` — C5 exclusive owner     | 0   | 0   | 0   | 225 | 0   | 0  | 225 |
| `client_5` — C6 redundancy control  | 25  | 25  | 25  | 0   | 0   | 25 | 100 |
| **Total train** | 225 | 225 | 225 | 225 | 225 | 224 | **1349** |
| Val (shared)    | 30  | 30  | 30  | 30  | 30  | 30  | 180 |
| Test (central)  | 45  | 45  | 45  | 45  | 45  | 45  | 270 |

The design labels these C1–C6; on disk they are the zero-based `client_0`–`client_5`
folders `client.py` expects, so **C5 (exclusive owner) is `client_4`**.

```bash
python utils/dataset_creation/split_neu_data.py \
    --src data/NEU-DET-SOURCE \
    --out data/neu6_data \
    --preset neu6
```

**Centralized validation.** Unlike `neu3`, all clients share one class-balanced
validation set (`data/neu6_data/val/`). No client training images are spent on
validation — each client dir receives a copy of the shared set, so
`client.py`'s per-round `val(split="val")` and its image count keep working
unchanged.

**Invariants**, enforced by `validate_preset()` on the config and re-checked by
`verify_output()` against the written files (both raise rather than `assert`, so
`-O` cannot strip them):
- every class sums to exactly 225 across clients, less any declared pool
  exclusion (so 224 for scratches);
- `pitted_surface` appears only in `client_4` — **including at the box level**.
  NEU-DET files each image under one class folder but its XML may annotate other
  classes, so an image outside `pitted_surface/` can still carry a pitted box.
  Exactly one does (`scratches_228.jpg`), and it is dropped from the pool
  entirely rather than relabelled — labels stay a faithful conversion of the
  source XML. The resulting shortfall is declared in `expected_pool_exclusions`
  and cross-checked against the data at runtime, so if the source ever changes
  the run fails instead of silently shorting a client;
- test/val are exactly 45/30 per class, disjoint from each other and from all
  client training data;
- clients share no training images; every image has a non-empty label.

---

## Experiment: Disruption-Aware FL

### Stage 0 — Pre-disruption FL (Rounds 1–10)
All three clients train collaboratively with FedAvg.

```bash
# SLURM
sbatch run.sh 10 5 fedavg data/neu_data 0 disruption_neu_fedavg_seed0

# Local — same arguments, just bash instead of sbatch
bash run.sh 10 5 fedavg data/neu_data 0 disruption_neu_fedavg_seed0
```

`run.sh` arguments: `ROUNDS EPOCHS STRATEGY DATA_DIR SEED EXP_NAME`. Launch it with
`sbatch` on SLURM or `bash` locally — same arguments either way (this applies to
every `sbatch run.sh …` example below). `run.sh` starts the server and all three
clients for you and forwards the shared seed to each.

**Seeds are inputs, not hardcoded.** `--seed` controls the random detection-head
initialization (server, broadcast to all clients) and YOLO's training randomness
(augmentation, dataloader order). `run.sh` forwards its `SEED` argument to the
server and all clients, and — when `EXP_NAME` is omitted — folds the seed into the
default experiment name (`disruption_neu_<strategy>_seed<seed>`) so runs with
different seeds don't overwrite each other. Defaults: training scripts use `0`, the
dataset split scripts use `42` (preserving the original split). To assess whether a
result is consistent rather than a lucky run, repeat the full experiment across
several seeds — see [Robustness: repeat across seeds](#robustness-repeat-across-seeds).

### Choosing the disruption round (timing analysis)

The global model converges fast (often by round ~4 at 2 local epochs), so the
disruption round `t*` and local-epoch count `E` should be justified by evidence.
Run FL with a few values of `E`, then analyze all runs together:

```bash
# E-sweep (run.sh: ROUNDS EPOCHS STRATEGY DATA_DIR SEED EXP_NAME)
for E in 1 2 5; do
  sbatch run.sh 10 $E fedavg data/neu_data 0 conv_E${E}_seed0
done

# After they finish: evaluate every round's logged global on the central test set
python -m shapley.convergence \
    --log_dirs experiments/conv_E1_seed0/fl/shapley_logs \
               experiments/conv_E2_seed0/fl/shapley_logs \
               experiments/conv_E5_seed0/fl/shapley_logs \
    --labels E1 E2 E5 --local_epochs 1 2 5 \
    --test_dir data/neu_data/test \
    --out_dir experiments/convergence_analysis \
    --per_round_shapley
```

Outputs: per-round mAP50 curves with the detected plateau round marked
(`convergence_curves.png`, incl. a compute-fair round×E panel), a recommended
`t*` per run (`convergence.json`), and — with `--per_round_shapley` — exact
per-client Shapley values at every round (`shapley_by_round_*.csv/png`) showing
when contributions stabilize. Plateau rule: no mAP50 improvement > `--min_delta`
(0.005) for `--patience` (2) consecutive rounds. Use the recommended `(E, t*)`
for the disruption experiment and the Shapley persistence analysis below.

### Stage 1 — Head-Only Adaptation (Rounds 11–15)
Client 2 initializes from the Round-10 global model. Backbone frozen, only detection head trained.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
    --mode    head_only --epochs 25 --lr 0.001 --seed 0 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

### Stage 2 — Neck + Head Fine-Tuning (Rounds 16–30)
Backbone still frozen; neck and head unfrozen.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/adaptation/head_only/weights/best.pt \
    --mode    neck_head --epochs 75 --lr 0.0001 --seed 0 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

### Stage 3 — Full Fine-Tuning (optional)
Only run if Stage 2 still underperforms. Risk of overfitting on Client 2's small dataset.

```bash
python scripts/train_centralized.py \
    --data    data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/adaptation/neck_head/weights/best.pt \
    --mode    full --epochs 20 --lr 0.00001 --seed 0 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
```

`scripts/train_centralized.py` takes `--seed` (default `0`) — use the same seed as
the FL run when repeating a full experiment. It's recorded in each run's
`train_config.json`.

### Adaptive stage selection (alternative to fixed Stages 1–3)

Instead of fixed epoch counts, escalate `head_only → neck_head` (optionally
`→ full`) when Client 2's **validation mAP50 plateaus** — no improvement
> `--min_delta` for `--patience` consecutive segments:

```bash
python -m adaptation.adaptive_finetune \
    --weights experiments/disruption_neu_fedavg_seed0/fl/final_model/client_2_final.pt \
    --data data/neu_data/client_2/data.yaml \
    --out_dir experiments/disruption_neu_fedavg_seed0/adaptive \
    --seg_epochs 5 --patience 2 --min_delta 0.005 --max_epochs 100
```

Training proceeds in short segments (per-stage LRs: 1e-3 / 1e-4 / 1e-5); the
controller decides continue / escalate / stop after each. `trace.json` records
every segment's val mAP50 and the switch epochs — the evidence for *when* the
escalation happened. Add `--stages head_only neck_head full` to allow full
fine-tuning as the final stage.

### Knowledge distillation with class-retention weights

Retain what the offline factories (A, B) taught the model — without their data.
A frozen teacher (the pre-disruption global) runs on C's batches and the
student's class predictions are pulled toward it, weighted per class by the
contribution the offline clients stand to lose (from the
[contribution matrix](#per-class-contribution-matrix)):

```bash
python -m adaptation.distill_finetune \
    --weights experiments/disruption_neu_fedavg_seed0/fl/final_model/client_2_final.pt \
    --weights_json experiments/disruption_neu_fedavg_seed0/contribution/class_retention_weights.json \
    --data data/neu_data/client_2/data.yaml \
    --out_dir experiments/disruption_neu_fedavg_seed0/distill \
    --mode neck_head --epochs 75 --lam 1.0 --temperature 2.0
```

Total loss: `box + cls + dfl + λ · Σ_c w_c · KD_c`. `--lam 0` reproduces a
plain fine-tune (sanity baseline); `--teacher_conf 0.25` restricts distillation
to anchors the teacher is confident about; `--adaptive` combines KD with the
adaptive stage controller. Verify the KD gain by re-running the persistence +
contribution analyses on the distilled trajectory (retention ρ_A, ρ_B should
improve for the high-weight classes).

---

## Robustness: repeat across seeds

To confirm a result is consistent rather than a lucky run, run the **entire**
disruption experiment end-to-end for several seeds and compare the spread of the
final metrics. `--seed` controls three independent sources of randomness:

| Source | Where the seed acts | Controlled by |
|--------|--------------------|---------------|
| **Data split** — which images land in each client/test split | `split_neu_data.py` shuffle | `--seed` on the split script |
| **Model init** — the random detection head the server builds and broadcasts to all clients in round 1 | `set_seed()` before `load_model` in `server.py` | `--seed` on `server.py` |
| **Training** — augmentation (flips, HSV, mosaic) + mini-batch order, every round | `seed=` passed into `model.train()` | `--seed` on `client.py` / `train_centralized.py` |

### Fixed data, varying training seed (recommended)

Split the dataset **once**, then vary only the model-init + training seed. Every
replicate sees byte-for-byte identical data, so the spread of final metrics is a
clean measure of *training* robustness. Point every run at the same `data/neu_data`.

```bash
# One-time: build the dataset (do NOT re-run per seed)
python utils/dataset_creation/split_neu_data.py --src NEU-DET --out data/neu_data --seed 42

DATA=data/neu_data
for SEED in 0 1 2 3 4; do
  EXP=disruption_neu_fedavg_seed${SEED}

  # Stage 0 — FL rounds 1–10  (run.sh: ROUNDS EPOCHS STRATEGY DATA_DIR SEED EXP_NAME)
  sbatch run.sh 10 5 fedavg $DATA $SEED $EXP

  # Stages 1–2 — adaptation (same seed, same data)
  python scripts/train_centralized.py --data $DATA/client_2/data.yaml \
      --weights experiments/$EXP/fl/final_model/client_0_final.pt \
      --mode head_only --epochs 25 --lr 0.001 --seed $SEED \
      --output_dir experiments/$EXP/adaptation
  python scripts/train_centralized.py --data $DATA/client_2/data.yaml \
      --weights experiments/$EXP/adaptation/head_only/weights/best.pt \
      --mode neck_head --epochs 75 --lr 0.0001 --seed $SEED \
      --output_dir experiments/$EXP/adaptation

  # Analyze this seed's checkpoints
  python scripts/analyze_results.py \
      --round10 experiments/$EXP/fl/final_model/client_0_final.pt \
      --stage1  experiments/$EXP/adaptation/head_only/weights/best.pt \
      --stage2  experiments/$EXP/adaptation/neck_head/weights/best.pt \
      --client2_data $DATA/client_2/data.yaml \
      --central_data $DATA/test/data.yaml \
      --output experiments/$EXP/analysis/results.json
done
```

Report the mean ± std of the final metrics across the per-seed `results.json` files.

### Also varying the data split

To additionally test robustness to *which* images each client gets, re-run the
split per seed into its own directory (`--out data/neu_data_seed${SEED}`, fixed
per-client counts) and point each run's `DATA_DIR` at it. This bundles all three
randomness sources into one replicate.

---

## Threshold Tuning

After training, find the optimal YOLO confidence threshold on the test set:

```bash
python utils/analysis/tune_threshold.py \
    --model  experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
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

## Contribution Persistence (Shapley)

Measures **how much of Factory A's and B's contribution to the model persists as
Factory C fine-tunes alone** after the disruption — as a decay curve over C's
fine-tuning checkpoints. Players are the 3 clients, so all `2³ = 8` coalitions are
enumerated and Shapley values are computed **exactly**. See
[docs/shapley_explained.md](docs/shapley_explained.md) for the full method
(intuition, diagrams, and worked examples).

**Step 1 — Run FL (logging is automatic).** `run.sh` passes `--log_dir` to the
server, so every run writes the Shapley inputs to
`experiments/<exp>/fl/shapley_logs/` (each round's per-client weights + image
counts, and a global checkpoint per round):

```bash
sbatch run.sh 10 5 fedavg data/neu_data 0 disruption_neu_fedavg_seed0
```

**Step 2 — Compute Shapley values + the retention curve.** Reconstructs the 8
coalition models at the disruption round `t*` (retrain-free, by re-aggregating the
logged updates), fine-tunes each on C, evaluates every checkpoint on the shared
test set, and computes exact Shapley per checkpoint:

```bash
python -m shapley.persistence \
    --log_dir experiments/disruption_neu_fedavg_seed0/fl/shapley_logs \
    --t_star  4 \
    --mode neck_head --epochs 60 --save_period 10 --seed 0 \
    --out_dir experiments/disruption_neu_fedavg_seed0/shapley
```

Pick an **early** `t*` (while the model is still improving) so contributions are
clearly nonzero — the retention ratio `ρ_i(τ) = φ_i(τ) / φ_i(t*)` is unstable when
`φ_i(t*) ≈ 0`. Outputs land in `--out_dir`:

| File | Contents |
|------|----------|
| `retention_curve.png` | `ρ_A`, `ρ_B` vs C's fine-tuning epochs — the headline plot |
| `shapley_by_checkpoint.csv` | `φ_A / φ_B / φ_C` at each checkpoint |
| `retention.csv` | `ρ_i(τ)` per checkpoint |
| `forgetting_per_class.csv` | per-class AP@50 over fine-tuning (the corroborating forgetting proxy) |
| `results.json` | everything above + config |

### Per-class contribution matrix

The manuscript's **knowledge contribution matrix** (factories × defect classes):
exact Shapley values computed per class from the same 8 coalition evaluations a
persistence run already performs (dumped to `records.json` — no extra training):

```bash
python -m shapley.contribution \
    --records experiments/disruption_neu_fedavg_seed0/shapley/records.json \
    --out_dir experiments/disruption_neu_fedavg_seed0/contribution \
    --offline 0 1 --weight_mode lost
```

Outputs: `contribution_matrix.csv/json` (φ per player × class × checkpoint),
`retention_matrix.csv` (per-class ρ), `contribution_heatmap.png`, and
`class_retention_weights.json` — the per-class KD weights. `--weight_mode lost`
weights classes by contribution the offline clients **lost** during C's
adaptation (φ at τ=0 minus φ at the last checkpoint); `static` uses the τ=0
contribution directly. A `--tau0_only` mode computes the static matrix straight
from FL logs without any fine-tuning:

```bash
python -m shapley.contribution --tau0_only \
    --log_dir experiments/disruption_neu_fedavg_seed0/fl/shapley_logs \
    --t_star 4 --test_dir data/neu_data/test \
    --out_dir experiments/disruption_neu_fedavg_seed0/contribution
```

**Notes.**
- `--num_classes` must match the FL run (default `3` on both sides). Per-class
  forgetting needs `nc ≥ 2`.
- Reconstruction assumes weighted-average aggregation, so it supports **`fedavg`
  and `fedprox` only** (identical server step). Runs made with `fedawa` /
  `adaptive` are refused with a clear error — their weighting differs and would
  give wrong Shapley values.
- The pure math (reconstruction, Shapley, retention assembly) is unit-tested:
  `python -m shapley.tests.test_shapley` (and `test_reconstruct`, `test_logger`,
  `test_persistence`, `test_convergence`, `test_contribution`; the adaptation
  package has `python -m adaptation.tests.test_controller`, `test_kd`, and
  `test_ultralytics_api` — the last one tripwires the pinned Ultralytics APIs
  the KD loss relies on).

---

## Baselines

Reuses `scripts/train_centralized.py` with different `--output_dir` targets.

**Centralized model** — all client data pooled, standard YOLO training. First
merge the client folders into one dataset, then train on it:
```bash
python utils/data/centralized_dataset.py   # writes data/neu_centralized/

python scripts/train_centralized.py \
    --data data/neu_centralized/data.yaml \
    --mode full --epochs 150 --lr 0.01 --seed 0 \
    --output_dir experiments/disruption_neu_fedavg/baselines/centralized
```

**Client-only models** — each factory trains independently, no federation:
```bash
for i in 0 1 2; do
  python scripts/train_centralized.py \
      --data data/neu_data/client_${i}/data.yaml \
      --mode full --epochs 150 --lr 0.01 --seed 0 \
      --output_dir experiments/disruption_neu_fedavg/baselines/client_only/client_${i}
done
```

**Full FL baseline** — standard 30-round FL, no disruption:
```bash
sbatch run.sh 30 5 fedavg data/neu_data 0 fl_full_baseline
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
