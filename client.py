import sys
from pathlib import Path
from datetime import datetime
import flwr as fl
from ultralytics import RTDETR
from model import load_model, get_parameters, set_parameters
from data import get_dataset_yaml
import torch


class RTDETRClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_dir: str, epochs: int = 1):
        self.cid = cid
        self.data_dir = data_dir
        self.epochs = epochs
        self.model = load_model()
        self.round = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = (Path.cwd() / "fl_runs" / timestamp).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Client {self.cid}] Output dir: {self.base_dir}")

    def _run_dir(self) -> Path:
        """Returns fl_runs/<timestamp>/round_<N>/client_<cid>/"""
        d = self.base_dir / f"round_{self.round:02d}" / f"client_{self.cid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.round += 1
        set_parameters(self.model, parameters)

        run_dir = self._run_dir()

        self.model.train(
            data=get_dataset_yaml(self.data_dir),
            epochs=self.epochs,
            imgsz=128,
            workers=0,
            verbose=False,
            exist_ok=True,
            device=self.device,
            project=str((Path.cwd() / "fl_runs" / self.base_dir.name / f"round_{self.round:02d}").resolve()),
            name=f"client_{self.cid}",
        )

        # Reload from saved checkpoint
        last_ckpt = Path.cwd() / "fl_runs" / self.base_dir.name / f"round_{self.round:02d}" / f"client_{self.cid}" / "weights" / "last.pt"
        self.model = RTDETR(str(last_ckpt))
        self.model.to(self.device)

        print(f"[Client {self.cid}] Round {self.round} train done → {run_dir}")
        return get_parameters(self.model), self._count_images("train"), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        metrics = self.model.val(
            data=get_dataset_yaml(self.data_dir),
            split="val",
            verbose=False,
            device=self.device,
            project=str((Path.cwd() / "fl_runs" / self.base_dir.name / f"round_{self.round:02d}").resolve()),
            name=f"client_{self.cid}_val"
        )

        map50   = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        print(f"[Client {self.cid}] Round {self.round} eval — mAP50: {map50:.4f} | mAP50-95: {map5095:.4f}")
        return map5095, self._count_images("test"), {
            "mAP50": map50,
            "mAP50-95": map5095,
        }

    def _count_images(self, split: str) -> int:
        img_dir = Path(self.data_dir) / "images" / split
        return len(list(img_dir.glob("*.jpg")))


def main():
    cid = sys.argv[1]
    data_dir = f"data/client_{cid}"
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=RTDETRClient(cid, data_dir),
    )

if __name__ == "__main__":
    main()