from pathlib import Path
import flwr as fl
from model import MODEL_PATH, load_model, get_parameters, set_parameters, set_seed
from data import get_dataset_yaml
import torch
import numpy as np
import json
import time
import argparse
import copy


class YOLOClient(fl.client.NumPyClient):
    def __init__(self, cid: str, data_dir: str, out_dir: str, epochs: int = 5, num_classes: int = 3, strategy: str = "fedavg", seed: int = 0):
        """Initializes a YOLO Model for client {cid}.

        Args:
            cid (str): client id
            data_dir (str): data directory
            out_dir (str): base output directory for FL round artifacts (e.g. experiments/<exp_name>/fl)
            epochs (int, optional): Number of epochs run locally. Defaults to 5.
            num_classes (int, optional): Number of detection classes. Defaults to 3.
            strategy (str, optional): Aggregation strategy used at Server level.
            seed (int, optional): Random seed forwarded to YOLO training (augmentation,
                dataloader order). Defaults to 0.
        """
        self.cid = cid
        self.data_dir = data_dir
        self.epochs = epochs
        self.num_classes = num_classes
        self.strategy = strategy
        self.seed = seed
        self.model = load_model(num_classes=num_classes)
        self.round = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Diagnostic: snapshot of the global weights set at the start of fit(),
        # used by the on_train_start hook to confirm train() actually trains
        # from the global weights (not a reloaded yolov8n.pt checkpoint).
        self._global_snapshot = None
        self.model.add_callback("on_train_start", self._verify_global_loaded_hook)

        self.base_dir = Path(out_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Client {self.cid}] Output dir: {self.base_dir}")

        # FedProx: the server (FedProx strategy) forwards μ per round in the fit
        # config; the proximal term (μ/2)||w - w_global||² MUST be applied
        # client-side — Flower's FedProx is byte-for-byte FedAvg on the server
        # and never adds it for us. Register the hook unconditionally and gate on
        # μ>0 at train start (see _fedprox_hook), so FedProx is driven by what the
        # server sends, not the client's --strategy flag. This avoids a silent
        # client/server strategy mismatch quietly degrading FedProx to FedAvg.
        self._fedprox_mu = 0.0
        self.model.add_callback("on_train_start", self._fedprox_hook)

    def _fedprox_hook(self, trainer):
        """Add the FedProx term (μ/2)||w - w_global||² to the local loss.

        No-op unless the server sent μ>0 this round, so it is safe to register
        for every strategy — only FedProx's configure_fit forwards proximal_mu.
        """
        mu = self._fedprox_mu
        if mu <= 0:
            return
        net = trainer.model
        anchor = [p.detach().clone() for p in net.parameters()]
        base_loss = net.loss

        def loss(batch, preds=None):
            out, items = base_loss(batch, preds)
            out = out.clone()
            out[0] += 0.5 * mu * sum(((p - a) ** 2).sum()
                                     for p, a in zip(net.parameters(), anchor))
            return out, items

        net.loss = loss
        print(f"[Client {self.cid}] FedProx active (mu={mu})")

    def _verify_global_loaded_hook(self, trainer):
        """Confirm train() starts from the GLOBAL aggregated weights set in fit().

        If Ultralytics silently reloaded yolov8n.pt instead of using the
        in-memory weights, the trainer's model here would NOT match the global
        weights we set. Decisive from round >= 2 (in round 1 the global weights
        equal the pretrained init, so a match is expected either way).
        """
        try:
            if self._global_snapshot is None:
                return
            cur = [t.detach().cpu().numpy() for t in trainer.model.state_dict().values()]
            if len(cur) != len(self._global_snapshot):
                print(f"[Client {self.cid}] on_train_start: tensor count mismatch "
                      f"({len(cur)} vs {len(self._global_snapshot)}) — cannot verify")
                return
            diff = sum(
                float(np.abs(c.astype(np.float64) - g.astype(np.float64)).sum())
                for c, g in zip(cur, self._global_snapshot)
                if np.issubdtype(c.dtype, np.floating)
            )
            status = ("OK: training from global weights"
                      if diff < 1e-3 else
                      "WARNING: trainer weights differ from global — train() may have "
                      "reloaded yolov8n.pt and discarded the aggregated weights")
            print(f"[Client {self.cid}] Round {self.round} on_train_start: "
                  f"|trainer_model - global_set|_L1 = {diff:.6e} ({status})")
        except Exception as e:
            print(f"[Client {self.cid}] on_train_start verify hook failed: {e}")

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
            # Snapshot the global weights so the on_train_start hook can verify
            # train() actually trains from these (not a reloaded checkpoint).
            self._global_snapshot = [np.copy(p) for p in parameters]

            # FedProx μ for this round (server forwards it via the fit config);
            # read each round so a server-side schedule on μ is respected.
            self._fedprox_mu = float(config.get("proximal_mu", 0.0))

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
                seed=self.seed,
            )

            params = get_parameters(self.model)

            # Get precision, recall, and map50 for adaptive weighting aggregation
            precision, recall, map50 = 0.0, 0.0, 0.0
            if self.strategy == "adaptive":
                val_model = load_model(num_classes=self.model.model.nc)
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
                # Report our launch cid so the server can label each logged update
                # unambiguously — the server-side ClientProxy.cid is a Flower node
                # id, not the "0/1/2" we were launched with (see shapley/logger.py).
                "cid": self.cid,
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
        set_parameters(self.model, parameters)

        # Create model for validation (b/c of layer fusing problem)
        val_model = load_model(num_classes=self.model.model.nc)
        val_model.model.load_state_dict(self.model.model.state_dict())

        metrics = val_model.val(
            data=get_dataset_yaml(self.data_dir),
            split="val",
            verbose=False,
            workers=0, 
            device=self.device,
            project=str(self.base_dir / f"round_{self.round:02d}"),
            name=f"client_{self.cid}_val"
        )

        del val_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        map50     = float(metrics.box.map50)
        map5095   = float(metrics.box.map)
        map75     = float(metrics.box.map75)
        precision = float(metrics.box.mp)
        recall    = float(metrics.box.mr)
        f1_arr    = metrics.box.f1
        f1        = float(np.mean(f1_arr)) if len(f1_arr) else 0.0
        infer_ms  = float(metrics.speed["inference"])

        results_dict = {
            "round":                    self.round,
            "mAP50":                    map50,
            "mAP50-95":                 map5095,
            "mAP75":                    map75,
            "precision":                precision,
            "recall":                   recall,
            "f1":                       f1,
            "inference_ms_per_image":   infer_ms,
        }
        metrics_path = (
            self.base_dir
            / f"round_{self.round:02d}"
            / f"client_{self.cid}_val"
            / "metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fh:
            json.dump(results_dict, fh, indent=2)

        print(
            f"[Client {self.cid}] Round {self.round} eval — "
            f"mAP50: {map50:.4f} | mAP50-95: {map5095:.4f} | mAP75: {map75:.4f} | "
            f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | "
            f"infer: {infer_ms:.1f}ms/img"
        )
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=3)
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
    parser.add_argument(
        "--out_dir",
        type=str,
        default="fl_runs",
        help="Base output directory for FL round artifacts (e.g. experiments/<exp_name>/fl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for YOLO training (augmentation, dataloader order). "
             "Use the same seed as the server for a reproducible run.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    time.sleep(int(args.cid) * 3)
    print(f"[Client {args.cid}] starting (seed={args.seed})...", flush=True)
    data_dir = str(Path(f"{args.data_dir}/client_{args.cid}").resolve())

    # pre-initialize client fully before connecting
    client = YOLOClient(args.cid, data_dir, out_dir=args.out_dir, epochs=args.epochs, num_classes=args.num_classes, strategy=args.strategy, seed=args.seed)
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