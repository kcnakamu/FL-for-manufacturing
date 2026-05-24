from pathlib import Path
from datetime import datetime
import flwr as fl
from model import MODEL_PATH, load_model, get_parameters, set_parameters
from data import get_dataset_yaml
import torch
import time
import argparse
import copy


class YOLOClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_dir: str, timestamp: str, epochs: int = 5, num_classes: int = 1, strategy: str = "fedavg"):
        """Initializes a YOLO Model for client {cid}.

        Args:
            cid (str): client id
            data_dir (str): data directory
            timestamp (str): time the experiment started; used for output folder naming
            epochs (int, optional): Number of epochs run locally. Defaults to 1.
            num_classes (int, optional): Number of detection classes. Defaults to 1.
            strategy (str, optional): Aggregation strategy used at Server level.
        """
        self.cid = cid
        self.data_dir = data_dir
        self.epochs = epochs
        self.num_classes = num_classes
        self.strategy = strategy
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
            # Update the model's parameters using the server parameters
            set_parameters(self.model, parameters)

            run_dir = self._run_dir()
            self.model.overrides.setdefault("model", MODEL_PATH)
            
            # Train the updated model using the client's training set
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

            # Get precision, recall, and map50 for adaptive weighting aggregation
            precision, recall, map50 = 0.0, 0.0, 0.0
            if self.strategy == "adaptive":
                val_model = load_model(num_classes=self.num_classes)
                val_model.model.load_state_dict(self.model.model.state_dict())
                val_metrics = val_model.val(
                    data=get_dataset_yaml(self.data_dir),
                    split="val",
                    verbose=False,
                    workers=0,
                    device=self.device,
                    project=str(self.base_dir / f"round_{self.round:02d}"),
                    name=f"client_{self.cid}_fit_val",
                    exist_ok=True,
                )
                precision = float(val_metrics.box.mp)
                recall    = float(val_metrics.box.mr)
                map50     = float(val_metrics.box.map50)
                del val_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print(
                f"[Client {self.cid}] Round {self.round} train done → {run_dir} "
                f"| P={precision:.4f} R={recall:.4f} mAP50={map50:.4f}"
            )
            return params, self._count_images("train"), {
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
            }

        except Exception as e:
            print(f"[Client {self.cid}] fit() crashed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate(self, parameters, config):
        """ Evaluates model with the new global model after aggregation.
        """
        if self.model is None:
            self.model = load_model(num_classes=self.num_classes)
        set_parameters(self.model, parameters)

        # Create model for validation (b/c of layer fusing problem)
        val_model = load_model(num_classes=self.num_classes)
        val_model.model.load_state_dict(self.model.model.state_dict())

        metrics = val_model.val(
            data=get_dataset_yaml(self.data_dir),
            split="val",
            verbose=False,
            workers=0, 
            device=self.device,
            project=str((Path.cwd() / "fl_runs" / self.base_dir.name / f"round_{self.round:02d}").resolve()),
            name=f"client_{self.cid}_val"
        )

        del val_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    parser.add_argument(
        "--strategy",
        choices=["adaptive", "fedawa", "fedavg", "fedprox"],
        default="fedavg",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory; each client uses <data_dir>/client_<cid>",
    )
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    time.sleep(int(args.cid) * 3)
    print(f"[Client {args.cid}] starting...", flush=True)
    data_dir = f"{args.data_dir}/client_{args.cid}"
    
    # pre-initialize client fully before connecting
    client = YOLOClient(args.cid, data_dir, timestamp=timestamp, epochs=args.epochs, num_classes=args.num_classes, strategy=args.strategy)
    print(f"[Client {args.cid}] ready, connecting to server...", flush=True)
    
    fl.client.start_numpy_client(
        server_address=f"{args.server_host}:8080",
        client=client,
    )

    # Save final global model weights (client 0 is canonical; others also save their copy)
    weights_dir = client.base_dir / "final_model"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"client_{args.cid}_final.pt"
    client.model.save(str(save_path))
    print(f"[Client {args.cid}] Final model saved → {save_path}")

if __name__ == "__main__":
    main()