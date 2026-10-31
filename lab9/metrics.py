"""Métricas en escala original (kWh), agregadas sobre todos los clientes y orígenes.

Todas reciben el DataFrame largo que devuelve `backtest`:
    unique_id, ds, cutoff, h, y, y_hat [, q0.1, q0.5, q0.9, ...], model
y la tabla `stats` con la escala del MASE de cada serie (MAE in-sample del seasonal naive m=24,
calculado solo con entrenamiento).
"""
import numpy as np
import pandas as pd

from . import config as C


def quantile_cols(df):
    return sorted([c for c in df.columns if c.startswith("q") and c[1:].replace(".", "").isdigit()],
                  key=lambda c: float(c[1:]))


def _prep(df, stats):
    out = df.merge(stats[["unique_id", "mase_scale"]], on="unique_id", how="left")
    out["err"] = out["y"] - out["y_hat"]
    out["abs_err"] = out["err"].abs()
    out["sq_err"] = out["err"] ** 2
    out["scaled_err"] = out["abs_err"] / out["mase_scale"]
    return out


def _agg(g):
    """Métricas de un grupo de filas ya preparado con _prep."""
    nz = g["y"].abs() > 0
    res = {
        "MAE": g["abs_err"].mean(),
        "RMSE": np.sqrt(g["sq_err"].mean()),
        "WAPE": g["abs_err"].sum() / g["y"].abs().sum(),
        # MAPE: indefinido con y = 0, así que esas filas se excluyen; con y cercano a 0 explota.
        "MAPE": (g.loc[nz, "abs_err"] / g.loc[nz, "y"].abs()).mean(),
        "MASE": g["scaled_err"].mean(),
    }
    qs = quantile_cols(g)
    if qs:
        losses = []
        for c in qs:
            q = float(c[1:])
            d = g["y"] - g[c]
            losses.append(np.maximum(q * d, (q - 1) * d).mean())
        res["pinball"] = float(np.mean(losses))
        if "q0.1" in g and "q0.9" in g:
            res["cov_P10_P90"] = ((g["y"] >= g["q0.1"]) & (g["y"] <= g["q0.9"])).mean()
    return pd.Series(res)


def summary(df, stats):
    """Una fila con MAE, RMSE, WAPE, MAPE y MASE (más pinball y cobertura si hay cuantiles)."""
    return _agg(_prep(df, stats))


def by(df, stats, key):
    """Métricas por grupo: 'h' (paso del horizonte), 'hour' (hora del día) o 'unique_id'."""
    p = _prep(df, stats)
    if key == "hour":
        p["hour"] = p["ds"].dt.hour
    cols = [c for c in p.columns if c not in (key,)]
    return p.groupby(key)[cols].apply(_agg).reset_index()


def xmas_week(df):
    return df[(df["ds"] >= C.XMAS_START) & (df["ds"] <= C.XMAS_END)]


def evaluate(df, stats, name=None, extra=None):
    """Resumen completo para la tabla comparativa: métricas globales + semana del 22-28 dic."""
    row = {"model": name or df["model"].iloc[0]}
    row.update(summary(df, stats).to_dict())
    xm = xmas_week(df)
    if len(xm):
        s = summary(xm, stats)
        row["MASE_xmas"], row["MAE_xmas"] = s["MASE"], s["MAE"]
    if extra:
        row.update(extra)
    return row
