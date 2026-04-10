# Federated Learning for Manufacturing Defect Detection

This project trains a YOLOv8 object detector with Federated Learning (Flower) across multiple simulated manufacturing clients.

Each client trains locally on its own dataset partition (`data/client_<id>`), sends model updates to a central Flower server, and receives aggregated global parameters for the next round.

## Project Structure

- `server.py` - Starts the Flower server and runs FedAvg aggregation.
- `client.py` - Runs one FL client that trains/evaluates YOLO locally.
- `model.py` - Loads YOLOv8n and handles parameter get/set.
- `data.py` - Provides the dataset YAML path for each client.
- `run.sh` - SLURM script to launch server + 3 clients on one node.
- `generate_report.py` - Compiles per-round client metrics and plots.
- `requirements.txt` - Python dependencies.

## Requirements

- Python 3.10+ recommended
- CUDA-capable GPU recommended (CPU fallback is supported)
- Python packages in `requirements.txt`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Layout

Each client expects data in:

```text
data/
  client_0/
    data.yaml
    images/
      train/*.jpg
      val/*.jpg
    labels/
      train/*.txt
      val/*.txt
  client_1/
    ...
  client_2/
    ...
```

`client.py` uses `data/client_<cid>/data.yaml` automatically.

## Run Locally (without SLURM)

1. Start server:

```bash
python server.py --rounds 10
```

2. In separate terminals, start clients:

```bash
python client.py 0 localhost <timestamp> --epochs 1
python client.py 1 localhost <timestamp> --epochs 1
python client.py 2 localhost <timestamp> --epochs 1
```

Use the same `<timestamp>` for all clients so outputs are grouped together under:

```text
fl_runs/<timestamp>/
```

## Run on SLURM

`run.sh` launches:

- 1 Flower server
- 3 clients (`client_0`, `client_1`, `client_2`)

Submit:

```bash
sbatch run.sh [ROUNDS] [EPOCHS]
```

Defaults:

- `ROUNDS=10`
- `EPOCHS=1`

Logs are written to timestamped files under `logs/`.

## Output Artifacts

Training artifacts are created in:

```text
fl_runs/<timestamp>/round_<xx>/client_<id>/
```

Each round also writes validation outputs for each client.

## Generate Training Report

After training:

```bash
python generate_report.py
```

Or for a specific run:

```bash
python generate_report.py 20260403_070643
```

This generates in the run directory:

- `fl_report.png` - aggregated and per-client metric plots
- `fl_summary.csv` - flattened per-round/per-client summary table

## Notes

- The model backbone is `yolov8n.pt` (downloaded automatically by Ultralytics if needed).
- `server.py` currently expects at least 3 available clients.
- Default number of classes is set to 6 in `model.py`.
