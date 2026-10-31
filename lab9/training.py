"""Entrenamiento de las redes propias con early stopping en validación y registro de iteraciones."""
import copy
import json
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from . import config as C
from .data import CAL_FEATURES, calendar_features
from .dataset import WindowDataset, collate_batch, make_inputs, norm_params
from .models import count_params

LOSSES = {"mae": nn.L1Loss, "mse": nn.MSELoss, "huber": nn.HuberLoss}
EVAL_BATCH = 1024


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hardware():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


class TorchForecaster:
    """Envoltorio fit/predict para cualquier red de lab9.models.

    build(L, n_feat, n_cal, H) -> nn.Module construye la red; los demás argumentos son los
    hiperparámetros de entrenamiento que se exploran en la sección 5.2.
    Cada epoch usa `samples_per_epoch` ventanas muestreadas al azar del bloque de entrenamiento
    (stride 1); la validación usa un origen por día a las 00:00, igual que el backtest.
    """

    def __init__(self, name, build, L=168, H=C.H, norm="series", batch_size=512, lr=1e-3,
                 weight_decay=0.0, max_epochs=30, patience=5, samples_per_epoch=200_000,
                 history_months=None, loss="mae", grad_clip=1.0, seed=C.SEED, strategy="multi-salida",
                 uses_cov=True, family="", device=None, verbose=True):
        self.name, self.build = name, build
        self.L, self.H, self.norm = L, H, norm
        self.batch_size, self.lr, self.weight_decay = batch_size, lr, weight_decay
        self.max_epochs, self.patience, self.samples_per_epoch = max_epochs, patience, samples_per_epoch
        self.history_months, self.loss_name, self.grad_clip, self.seed = history_months, loss, grad_clip, seed
        self.strategy, self.uses_cov, self.family = strategy, uses_cov, family
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    # ------------------------------------------------------------------
    def config(self):
        arch = dict(getattr(self.build, "keywords", {}) or {})
        return {"model": self.name, "family": self.family, "strategy": self.strategy,
                "uses_cov": self.uses_cov, "L": self.L, "H": self.H, "norm": self.norm,
                "history_months": self.history_months if self.history_months else 30,
                "batch_size": self.batch_size, "lr": self.lr, "weight_decay": self.weight_decay,
                "loss": self.loss_name, "max_epochs": self.max_epochs, "patience": self.patience,
                "samples_per_epoch": self.samples_per_epoch, "seed": self.seed, **arch}

    def _val_epoch(self, loader, crit):
        self.net.eval()
        tot, n, abs_err, scaled_err = 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for x, xf, y, loc, scale, sid in loader:
                x, xf, y = x.to(self.device), xf.to(self.device), y.to(self.device)
                out = self.net(x, xf)
                tot += crit(out, y).item() * len(y)
                n += len(y)
                e = ((out - y).cpu() * scale[:, None]).abs()             # error en kWh
                abs_err += e.sum().item()
                scaled_err += (e / self.mase_scale[sid][:, None]).sum().item()
        return tot / n, abs_err / (n * self.H), scaled_err / (n * self.H)

    def fit(self, panel, val_start):
        set_seed(self.seed)
        val_start = pd.Timestamp(val_start)
        train_end = val_start - pd.Timedelta(hours=1)
        train_panel = panel.until(train_end)
        # Normalización y escala del MASE: SOLO con datos de entrenamiento.
        self.loc, self.scale = norm_params(train_panel, self.norm)
        tr = train_panel.Y
        self.mase_scale = torch.as_tensor(np.abs(tr[:, C.SEASON:] - tr[:, :-C.SEASON]).mean(1),
                                          dtype=torch.float32)
        train_ds = WindowDataset(panel, self.L, self.H, panel.times[0], train_end, self.loc, self.scale,
                                 self.norm, stride=1, history_months=self.history_months)
        val_ds = WindowDataset(panel, self.L, self.H, val_start, panel.times[-1], self.loc, self.scale,
                               self.norm, stride=24)
        self.n_train_windows, self.n_val_windows = len(train_ds), len(val_ds)
        gen = torch.Generator().manual_seed(self.seed)
        n_samples = min(self.samples_per_epoch, len(train_ds))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, collate_fn=collate_batch,
                              sampler=RandomSampler(train_ds, replacement=True, num_samples=n_samples,
                                                    generator=gen))
        # Batch de evaluación moderado: con LSTM y batch 4096 cuDNN reserva ~7.5 GB en inferencia.
        val_dl = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_batch)

        self.net = self.build(self.L, train_ds.n_features, len(CAL_FEATURES), self.H).to(self.device)
        self.n_params, self.n_trainable = count_params(self.net)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = LOSSES[self.loss_name]()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        self.history, best, best_state, bad = [], np.inf, None, 0
        t_start = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            run, n = 0.0, 0
            for x, xf, y, _, _, _ in train_dl:
                x, xf, y = x.to(self.device), xf.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = crit(self.net(x, xf, y), y)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                opt.step()
                run += loss.item() * len(y)
                n += len(y)
            val_loss, val_mae, val_mase = self._val_epoch(val_dl, crit)
            dt = time.perf_counter() - t0
            self.history.append({"epoch": epoch, "train_loss": run / n, "val_loss": val_loss,
                                 "val_MAE": val_mae, "val_MASE": val_mase, "epoch_time_s": dt})
            if self.verbose:
                print(f"  ep {epoch:02d} | train {run / n:.4f} | val {val_loss:.4f} | "
                      f"MAE {val_mae:.2f} kWh | MASE {val_mase:.3f} | {dt:.1f}s")
            if val_loss < best - 1e-5:
                best, best_state, bad = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.train_time = time.perf_counter() - t_start
        self.net.load_state_dict(best_state)
        self.gpu_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if self.device == "cuda" else 0.0
        hist = pd.DataFrame(self.history)
        self.best_epoch = int(hist.loc[hist["val_loss"].idxmin(), "epoch"])
        self.best_val = hist.loc[hist["val_loss"].idxmin()].to_dict()
        return self

    @torch.no_grad()
    def predict(self, y_hist, times_hist, times_future):
        self.net.eval()
        x, xf, loc, scale = make_inputs(y_hist, times_hist, times_future, self.L, self.loc, self.scale,
                                        self.norm, calendar_features)
        out = []
        for i in range(0, len(x), EVAL_BATCH):
            out.append(self.net(x[i:i + EVAL_BATCH].to(self.device), xf[i:i + EVAL_BATCH].to(self.device)).cpu())
        out = torch.cat(out)
        return (out * scale[:, None] + loc[:, None]).numpy()

    def summary(self):
        hist = pd.DataFrame(self.history)
        return {**self.config(), "params_total": self.n_params, "params_trainable": self.n_trainable,
                "train_windows": self.n_train_windows, "epochs_run": len(hist),
                "best_epoch": self.best_epoch, "train_time_s": self.train_time,
                "epoch_time_s": hist["epoch_time_s"].mean(), "gpu_peak_mb": self.gpu_peak_mb,
                "val_loss": self.best_val["val_loss"], "val_MAE": self.best_val["val_MAE"],
                "val_MASE": self.best_val["val_MASE"], "hardware": hardware()}


def save_run(model, info=None, path=None):
    """Agrega una fila a results/runs.csv y guarda la curva por epoch en results/history/."""
    C.ensure_dirs()
    path = path or C.RESULTS_DIR / "runs.csv"
    run_id = f"{model.name}_{datetime.now():%Y%m%d_%H%M%S}"
    row = {"run_id": run_id, **model.summary()}
    if info:
        row["infer_time_s"] = info.get("infer_time_s")
        row["infer_split"] = info.get("split")
    pd.DataFrame(model.history).to_csv(C.RESULTS_DIR / "history" / f"{run_id}.csv", index=False)
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return run_id
