"""Protocolo único de evaluación: backtesting con origen móvil diario.

Contrato que cumple todo modelo (propio o envoltorio de librería):

    model.name : str
    model.fit(panel, val_start)            -> panel ya truncado; val_start marca el bloque de
                                              validación (early stopping) dentro de ese panel
    model.predict(y_hist, times_hist, times_future)
        y_hist      : (N, t) kWh, SOLO datos anteriores al origen
        times_hist  : marcas de esas t horas
        times_future: las H horas a pronosticar (para covariables de calendario)
        -> ndarray (N, H) en kWh, o dict {"mean": (N, H), "q0.1": (N, H), ...}
"""
import time

import numpy as np
import pandas as pd

from . import config as C


def daily_origins(start, end, H=C.H):
    """Orígenes a las 00:00 de cada día cuyo horizonte completo cabe en [start, end]."""
    days = pd.date_range(start, end, freq="D")
    return [d for d in days if d + pd.Timedelta(hours=H - 1) <= end]


SPLITS = {"val": (C.VAL_START, C.VAL_END), "test": (C.TEST_START, C.TEST_END)}


def _to_long(pred, ids, times_future, cutoff, y_true):
    if not isinstance(pred, dict):
        pred = {"mean": pred}
    N, H = pred["mean"].shape
    df = pd.DataFrame({
        "unique_id": np.repeat(ids, H),
        "ds": np.tile(times_future, N),
        "cutoff": cutoff,
        "h": np.tile(np.arange(1, H + 1), N),
        "y": y_true.reshape(-1),
        "y_hat": np.asarray(pred["mean"]).reshape(-1),
    })
    for k, v in pred.items():
        if k != "mean":
            df[k] = np.asarray(v).reshape(-1)
    return df


def backtest(model, panel, split="test", H=C.H, fit=True, refit_every=None, val_months=3,
             verbose=True):
    """Evalúa `model` con origen móvil diario sobre el bloque `split` ('val' o 'test').

    - fit=True entrena UNA vez con panel.until(VAL_END) (entrenamiento + validación para early
      stopping). La prueba nunca se entrega a fit().
    - refit_every='MS' reentrena al inicio de cada mes con ventana expandible: en el origen o
      se llama fit(panel.until(o - 1h), val_start=o - val_months meses).
    - En cada origen el modelo solo recibe y_hist = Y[:, :origen] (datos hasta las 23:00 del día
      anterior) y las marcas de las H horas futuras.
    Devuelve (preds, info): preds en formato largo y info con tiempos de entrenamiento/inferencia.
    """
    start, end = SPLITS[split]
    origins = daily_origins(start, end, H)
    info = {"model": model.name, "split": split, "n_origins": len(origins), "fit_time_s": 0.0,
            "infer_time_s": 0.0, "n_fits": 0}

    def _fit(cutoff, val_start):
        t0 = time.perf_counter()
        model.fit(panel.until(cutoff), val_start)
        info["fit_time_s"] += time.perf_counter() - t0
        info["n_fits"] += 1

    if fit and refit_every is None:
        _fit(C.VAL_END, C.VAL_START)

    def _inputs(o):
        pos = panel.idx(o)
        # Todo lo anterior al origen y nada más; más las marcas de las H horas futuras.
        return panel.Y[:, :pos], panel.times[:pos], panel.times[pos:pos + H]

    frames = []
    if hasattr(model, "predict_many") and refit_every is None:
        # Modelos locales de StatsForecast: mismos insumos por origen, un solo pool de procesos.
        inputs = [_inputs(o) for o in origins]
        t0 = time.perf_counter()
        preds_all = model.predict_many(inputs)
        info["infer_time_s"] += time.perf_counter() - t0
        for o, (_, _, tf), pred in zip(origins, inputs, preds_all):
            pos = panel.idx(o)
            frames.append(_to_long(pred, panel.ids, tf, o - pd.Timedelta(hours=1), panel.Y[:, pos:pos + H]))
    else:
        for o in origins:
            if refit_every is not None and (o == origins[0] or o.is_month_start):
                cutoff = o - pd.Timedelta(hours=1)
                _fit(cutoff, o - pd.DateOffset(months=val_months))
            y_hist, times_hist, times_future = _inputs(o)
            t0 = time.perf_counter()
            pred = model.predict(y_hist, times_hist, times_future)
            info["infer_time_s"] += time.perf_counter() - t0
            pos = panel.idx(o)
            frames.append(_to_long(pred, panel.ids, times_future, o - pd.Timedelta(hours=1),
                                   panel.Y[:, pos:pos + H]))
    preds = pd.concat(frames, ignore_index=True)
    preds["model"] = model.name
    if verbose:
        print(f"[{model.name}] {split}: {len(origins)} orígenes | fit {info['fit_time_s']:.1f}s "
              f"| inferencia {info['infer_time_s']:.1f}s")
    return preds, info
