"""Modelos de librería con el mismo contrato fit/predict de backtest(): NeuralForecast y Chronos-Bolt."""
import copy
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import CAL_FEATURES, calendar_features
from .training import hardware

# Lightning imprime varias líneas en cada fit/predict; en el backtest se llama 92 veces. Se importa
# antes de bajar el nivel de sus loggers porque al importarse los vuelve a poner en INFO.
import pytorch_lightning  # noqa: E402,F401

for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*predict_dataloader.*")

TRAINER_KW = dict(enable_progress_bar=False, enable_model_summary=False, logger=False,
                  enable_checkpointing=False, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                  devices=1)


class BestWeights(pytorch_lightning.Callback):
    """Guarda en memoria los pesos con la menor pérdida de validación.

    El early stopping de NeuralForecast solo detiene el entrenamiento: se queda con los pesos del último
    paso, que pueden estar `patience` chequeos después del mejor. Este callback permite restaurarlos, igual
    que TorchForecaster con las redes propias.
    """

    def __init__(self):
        self.best, self.step, self.state = np.inf, None, None

    def on_validation_end(self, trainer, pl_module):
        v = trainer.callback_metrics.get("ptl/val_loss")
        if not trainer.sanity_checking and v is not None and np.isfinite(float(v)) and float(v) < self.best:
            self.best, self.step = float(v), trainer.global_step
            self.state = {k: t.detach().clone() for k, t in pl_module.state_dict().items()}


_Y_ONLY = {}


def scale_y_only(cls):
    """Subclase de un modelo de NeuralForecast que normaliza por ventana solo la serie objetivo.

    NeuralForecast normaliza también las covariables con la media y la desviación de la ventana de
    entrada. El seno/coseno del mes es constante dentro de una semana: su desviación sale ~1e-8 por
    redondeo (no exactamente 0, que NF sí reemplaza por 1), así que el día que cambia el mes la
    covariable se divide entre ~1e-6 y la red recibe valores de ~1e5 (pronósticos de ±50,000 kWh el
    1 de octubre o el 1 de diciembre). El calendario ya está en [-1, 1] y {0, 1}: no hace falta escalarlo.
    """
    if cls not in _Y_ONLY:
        _Y_ONLY[cls] = type(cls.__name__, (cls,), {"_get_temporal_exogenous_cols": lambda self, temporal_cols: []})
    return _Y_ONLY[cls]


def to_long(Y, times, ids, exog=True):
    """Matriz (N, t) -> formato largo de Nixtla (unique_id, ds, y [+ calendario])."""
    times = pd.DatetimeIndex(times)
    df = pd.DataFrame({"unique_id": np.repeat(ids, len(times)), "ds": np.tile(times.values, len(ids)),
                       "y": np.asarray(Y, dtype=np.float32).reshape(-1)})
    if exog:
        cal = calendar_features(times).to_numpy(np.float32)
        df[CAL_FEATURES] = np.tile(cal, (len(ids), 1))
    return df


class NFModel:
    """Envuelve un modelo de NeuralForecast con el contrato fit/predict de backtest().

    - El modelo se construye de nuevo en cada fit() (clase + hiperparámetros), así el reentrenamiento
      mensual parte de pesos nuevos y la semilla se controla con `random_seed`.
    - fit() recibe el panel ya truncado; las últimas horas desde `val_start` son el bloque de
      validación de NeuralForecast (`val_size`), que solo se usa para early stopping. Al terminar se
      restauran los pesos del mejor chequeo de validación (BestWeights).
    - step_size=24: las ventanas de entrenamiento y validación avanzan un día y terminan a las 23:00,
      igual que los orígenes del backtest. Además baja la validación de ~690k a ~29k ventanas.
    - predict() entrega a NeuralForecast solo las últimas L horas (y su calendario) y el calendario de
      las H horas futuras. La normalización es por ventana (scaler temporal o RevIN), sin fuga, y
      solo se aplica a la serie objetivo (ver scale_y_only).
    - Si el modelo es probabilístico (DistributionLoss con level=[80]) devuelve P10, P50 y P90; el
      pronóstico puntual es la mediana, que es lo que minimiza el MAE.
    """

    def __init__(self, name, cls, L=168, H=C.H, exog=True, history_months=None, family="", strategy="",
                 seed=C.SEED, **kw):
        self.name, self.cls, self.L, self.H, self.exog = name, cls, L, H, exog
        self.history_months, self.family, self.strategy, self.seed = history_months, family, strategy, seed
        self.uses_cov = exog
        self.kw = dict(kw)
        self.kw.setdefault("step_size", 24)

    def config(self):
        hp = {k: v for k, v in self.kw.items() if isinstance(v, (int, float, str, bool))}
        if "loss" in self.kw:
            hp["loss"] = type(self.kw["loss"]).__name__
        return {"model": self.name, "family": self.family, "strategy": self.strategy, "uses_cov": self.uses_cov,
                "L": self.L, "H": self.H, "history_months": self.history_months or 30, "seed": self.seed,
                "nf_class": self.cls.__name__, **hp}

    def _build(self):
        extra = {"futr_exog_list": list(CAL_FEATURES)} if self.exog else {}
        return scale_y_only(self.cls)(h=self.H, input_size=self.L, alias=self.name, random_seed=self.seed,
                                      callbacks=[BestWeights()], **extra, **copy.deepcopy(self.kw), **TRAINER_KW)

    def fit(self, panel, val_start):
        from neuralforecast import NeuralForecast
        val_pos = panel.idx(val_start)
        start = 0
        if self.history_months:      # historia de entrenamiento limitada (sección 5.2)
            first = pd.Timestamp(val_start) - pd.DateOffset(months=self.history_months)
            start = max(0, panel.idx(first) - self.L)
        self.ids = list(panel.ids)
        df = to_long(panel.Y[:, start:], panel.times[start:], self.ids, self.exog)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        self.nf = NeuralForecast(models=[self._build()], freq="h")
        self.nf.fit(df=df, val_size=len(panel.times) - val_pos)
        self.train_time = time.perf_counter() - t0
        self.gpu_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        self.model = self.nf.models[0]
        # NeuralForecast copia el modelo al entrenar: el callback que se usó está en su trainer_kwargs.
        cbs = self.model.trainer_kwargs.get("callbacks") or []
        best = next((cb for cb in cbs if isinstance(cb, BestWeights)), None)
        self.best_step = None
        if best is not None and best.state is not None:
            self.model.load_state_dict(best.state)
            self.best_step = best.step
            self.model.trainer_kwargs["callbacks"] = [cb for cb in cbs if cb is not best]
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.history = self._history()
        return self

    def _history(self):
        """Curvas de pérdida: una fila por chequeo de validación (cada `val_check_steps` pasos)."""
        tr = pd.DataFrame(self.model.train_trajectories, columns=["step", "train_loss"])
        va = pd.DataFrame(self.model.valid_trajectories, columns=["step", "val_loss"])
        if va.empty:
            return tr.assign(val_loss=np.nan).to_dict("records")
        tr["bin"] = np.searchsorted(va["step"].values, tr["step"].values)
        va["train_loss"] = tr.groupby("bin")["train_loss"].mean().reindex(range(len(va))).values
        return va[["step", "train_loss", "val_loss"]].to_dict("records")

    def predict(self, y_hist, times_hist, times_future):
        L = min(self.L, y_hist.shape[1])
        hist = to_long(y_hist[:, -L:], times_hist[-L:], self.ids, self.exog)
        futr = None
        if self.exog:
            futr = to_long(np.zeros((len(self.ids), len(times_future))), times_future, self.ids).drop(columns="y")
        fc = self.nf.predict(df=hist, futr_df=futr)
        fc = fc.set_index(["unique_id", "ds"]).loc[pd.MultiIndex.from_product([self.ids, times_future])]
        pick = lambda col: fc[col].to_numpy(np.float32).reshape(len(self.ids), len(times_future))
        if f"{self.name}-lo-80" in fc:      # el intervalo central de 80% va de P10 a P90
            med = pick(f"{self.name}-median")
            return {"mean": med, "q0.1": pick(f"{self.name}-lo-80"), "q0.5": med,
                    "q0.9": pick(f"{self.name}-hi-80")}
        return pick(self.name)

    def summary(self):
        hist = pd.DataFrame(self.history)
        best = hist.loc[hist["val_loss"].idxmin()] if hist["val_loss"].notna().any() else None
        return {**self.config(), "params_total": self.n_params, "params_trainable": self.n_trainable,
                "steps_run": int(self.model.train_trajectories[-1][0]) + 1 if self.model.train_trajectories else np.nan,
                "best_step": self.best_step if self.best_step is not None else np.nan,
                "train_time_s": self.train_time, "gpu_peak_mb": self.gpu_peak_mb,
                "val_loss": best["val_loss"] if best is not None else np.nan, "hardware": hardware()}


class ChronosBolt:
    """Chronos-Bolt zero-shot: no se entrena; fit() no hace nada y predict() solo hace inferencia.

    Recibe las últimas `context` horas de cada serie (2048 es el contexto máximo de Bolt) y devuelve
    los cuantiles P10, P50 y P90; el pronóstico puntual es la mediana.
    """

    def __init__(self, name="ChronosBolt", model_id="amazon/chronos-bolt-small", context=2048,
                 batch_size=256, device=None):
        from chronos import BaseChronosPipeline
        self.name, self.model_id, self.context, self.batch_size = name, model_id, context, batch_size
        self.family, self.strategy, self.uses_cov, self.L = "Fundacional", "multi-salida (directa)", False, context
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        params = list(self.pipeline.model.parameters())
        self.n_params = self.n_trainable = sum(p.numel() for p in params)
        self.train_time = 0.0

    def fit(self, panel, val_start=None):
        return self

    @torch.no_grad()
    def predict(self, y_hist, times_hist, times_future):
        ctx = torch.as_tensor(np.asarray(y_hist[:, -self.context:]), dtype=torch.float32)
        # Más allá de 64 pasos Bolt realimenta su propia mediana y la memoria crece mucho: con H = 168 un lote
        # de 256 series pide ~7 GB (y en una GPU de 8 GB Windows desborda a la RAM); con 64 bastan ~2 GB.
        bs = self.batch_size if len(times_future) <= 64 else min(self.batch_size, 64)
        qs = []
        for i in range(0, len(ctx), bs):
            q, _ = self.pipeline.predict_quantiles(ctx[i:i + bs], prediction_length=len(times_future),
                                                   quantile_levels=[0.1, 0.5, 0.9])
            qs.append(q.float().cpu())
        q = torch.cat(qs).numpy()                                              # (N, H, 3)
        return {"mean": q[..., 1], "q0.1": q[..., 0], "q0.5": q[..., 1], "q0.9": q[..., 2]}
