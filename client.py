import sys
from pathlib import Path
from datetime import datetime
import flwr as fl
from ultralytics import YOLO
from model import load_model, get_parameters, set_parameters
from data import get_dataset_yaml
import torch
import time
import argparse


class YOLOClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_dir: str, timestamp: str, epochs: int = 1, num_classes: int = 2):
        self.cid = cid
        self.data_dir = data_dir
        self.epochs = epochs
        self.num_classes = num_classes 
        self.model = load_model(num_classes=num_classes) 
        self.round = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = (Path.cwd() / "fl_runs" / timestamp).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Client {self.cid}] Output dir: {self.base_dir}")

    def _run_dir(self) -> Path:
        d = self.base_dir / f"round_{self.round:02d}" / f"client_{self.cid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        try:
            self.round += 1
            
            new_model = load_model(num_classes=self.num_classes)
            
            # Restore local BN stats if available
            if hasattr(self, 'local_bn_stats'):
                new_model.model.load_state_dict(self.local_bn_stats, strict=False)
            
            self.model = new_model
            set_parameters(self.model, parameters)

            run_dir = self._run_dir()

            self.model.train(
                data=get_dataset_yaml(self.data_dir),
                epochs=self.epochs,
                imgsz=480,
                batch=16,
                workers=0,
                verbose=False,
                exist_ok=True,
                device=self.device,
                project=str(self.base_dir / f"round_{self.round:02d}"),
                name=f"client_{self.cid}",
                amp=True,
            )

            params = get_parameters(self.model)
            self.local_bn_stats = {k: v for k, v in self.model.model.state_dict().items() if "bn" in k.lower()}

            print(f"[Client {self.cid}] Round {self.round} train done → {run_dir}")
            return params, self._count_images("train"), {}

        except Exception as e:
            print(f"[Client {self.cid}] fit() crashed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, parameters, config):
        if self.model is None:
            self.model = load_model()
            set_parameters(self.model, parameters)

        metrics = self.model.val(
            data=get_dataset_yaml(self.data_dir),
            split="val",
            verbose=False,
            workers=0, 
            device=self.device,
            project=str((Path.cwd() / "fl_runs" / self.base_dir.name / f"round_{self.round:02d}").resolve()),
            name=f"client_{self.cid}_val"
        )

        map50   = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        print(f"[Client {self.cid}] Round {self.round} eval — mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}")
        return map5095, self._count_images("val"), {
            "mAP50": map50,
            "mAP50-95": map5095,
        }

    def _count_images(self, split: str) -> int:
        img_dir = Path(self.data_dir) / "images" / split
        return len(list(img_dir.glob("*.jpg")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cid", type=str)
    parser.add_argument("server_host", type=str, default="localhost")
    parser.add_argument("timestamp", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=1)
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    time.sleep(int(args.cid) * 3)
    print(f"[Client {args.cid}] starting...", flush=True)
    data_dir = f"data/client_{args.cid}"
    
    # pre-initialize client fully before connecting
    client = YOLOClient(args.cid, data_dir, timestamp=timestamp, epochs=args.epochs, num_classes=args.num_classes)
    print(f"[Client {args.cid}] ready, connecting to server...", flush=True)
    
    fl.client.start_numpy_client(
        server_address=f"{args.server_host}:8080",
        client=client,
    )

if __name__ == "__main__":
    main()