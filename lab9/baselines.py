"""Baselines estadísticos (StatsForecast) y LightGBM global, con el mismo contrato fit/predict."""
import time

import numpy as np
import pandas as pd

from . import config as C
from .data import calendar_features


class StatsForecastModel:
    """Modelo local de StatsForecast reajustado en cada origen con la historia disponible.

    `input_size` limita la historia que se le entrega (las últimas horas): Naive y seasonal
    naive solo necesitan la última semana; MSTL usa unas semanas para estimar la estacionalidad
    diaria y semanal.
    """

    def __init__(self, name, sf_model, input_size=24 * 7 * 2, n_jobs=1):
        self.name, self.sf_model, self.input_size, self.n_jobs = name, sf_model, input_size, n_jobs
        self.n_params = 0

    def fit(self, panel, val_start=None):
        return self

    def predict(self, y_hist, times_hist, times_future):
        return self.predict_many([(y_hist, times_hist, times_future)])[0]

    def predict_many(self, inputs):
        """Pronostica varios orígenes en una sola llamada a StatsForecast.

        Cada par (origen k, serie i) se vuelve una serie independiente con id k*N + i que contiene
        solo las `input_size` horas previas a ese origen. Como los modelos son locales, el
        resultado es idéntico a llamar predict() origen por origen, pero se reparte en un único
        pool de procesos (n_jobs) en vez de crear uno por origen.
        """
        from statsforecast import StatsForecast
        N, H = inputs[0][0].shape[0], len(inputs[0][2])
        frames = []
        for k, (y_hist, times_hist, _) in enumerate(inputs):
            y = y_hist[:, -self.input_size:]
            t = times_hist[-self.input_size:]
            frames.append(pd.DataFrame({"unique_id": np.repeat(np.arange(N) + k * N, len(t)),
                                        "ds": np.tile(t, N), "y": y.reshape(-1)}))
        df = pd.concat(frames, ignore_index=True)
        sf = StatsForecast(models=[self.sf_model], freq="h", n_jobs=self.n_jobs)
        fc = sf.forecast(df=df, h=H).sort_values(["unique_id", "ds"])
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return fc[col].to_numpy(np.float32).reshape(len(inputs), N, H)


# ---------------------------------------------------------------------------
# LightGBM global recursivo
# ---------------------------------------------------------------------------
LAGS = list(range(1, 25)) + [168]


def _cal_cols(times):
    times = pd.DatetimeIndex(times)
    hol = calendar_features(times)["holiday"].values
    return np.stack([times.hour, times.dayofweek, times.month, hol], axis=1).astype(np.float32)


CAL_NAMES = ["hour", "dow", "month", "holiday"]


class LightGBMGlobal:
    """Un solo LightGBM para todas las series, con rezagos 1-24 y 168 más calendario.

    - Objetivo escalado por la media de entrenamiento de cada serie (y / media) para que un
      modelo global sirva a clientes con escalas muy distintas.
    - Estrategia recursiva: predice t+1, agrega la predicción a la historia y repite 24 veces.
    - Se entrena con filas cuyo objetivo es anterior a val_start; el bloque de validación solo se
      usa para early stopping.
    """

    def __init__(self, name="LightGBM", history_months=None, max_rows=2_000_000, params=None,
                 num_boost_round=2000, early_stopping=50, seed=C.SEED):
        self.name = name
        self.history_months = history_months
        self.max_rows = max_rows
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.seed = seed
        self.params = {"objective": "l2", "learning_rate": 0.05, "num_leaves": 127,
                       "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "verbose": -1, "seed": seed, "num_threads": 0}
        if params:
            self.params.update(params)

    def _features(self, Ys, cal, t_idx):
        """Filas (serie, t) -> matriz [rezagos..., calendario]. Ys: (N, T) escalado."""
        lags = np.stack([Ys[:, t_idx - l] for l in LAGS], axis=-1)            # (N, n_t, n_lags)
        n, m = lags.shape[:2]
        calf = np.broadcast_to(cal[t_idx][None], (n, m, cal.shape[1]))
        return np.concatenate([lags, calf], axis=-1).reshape(n * m, -1)

    def _rows(self, Ys, cal, t_from, t_to, max_rows, rng):
        t_idx = np.arange(max(t_from, max(LAGS)), t_to)
        if len(t_idx) * Ys.shape[0] > max_rows:            # submuestreo uniforme de horas
            t_idx = np.sort(rng.choice(t_idx, size=max_rows // Ys.shape[0], replace=False))
        X = self._features(Ys, cal, t_idx)
        y = Ys[:, t_idx].reshape(-1)
        return X, y

    def fit(self, panel, val_start):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        val_pos = panel.idx(val_start)
        self.scale = panel.Y[:, :val_pos].mean(axis=1) + 1e-6          # solo entrenamiento
        Ys = panel.Y / self.scale[:, None]
        cal = _cal_cols(panel.times)
        t_from = 0
        if self.history_months is not None:
            t_from = int(np.searchsorted(panel.times, pd.Timestamp(val_start) -
                                         pd.DateOffset(months=self.history_months)))
        t0 = time.perf_counter()
        Xtr, ytr = self._rows(Ys, cal, t_from, val_pos, self.max_rows, rng)
        Xva, yva = self._rows(Ys, cal, val_pos, Ys.shape[1], 300_000, rng)
        names = [f"lag{l}" for l in LAGS] + CAL_NAMES
        dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=["dow"], free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        self.evals = {}
        self.booster = lgb.train(self.params, dtr, self.num_boost_round, valid_sets=[dtr, dva],
                                 valid_names=["train", "val"],
                                 callbacks=[lgb.early_stopping(self.early_stopping, verbose=False),
                                            lgb.record_evaluation(self.evals)])
        self.train_time = time.perf_counter() - t0
        self.n_rows = len(ytr)
        self.n_params = int(self.booster.num_trees())       # árboles; LightGBM no tiene "pesos"
        return self

    def predict(self, y_hist, times_hist, times_future):
        H = len(times_future)
        buf = np.concatenate([y_hist[:, -max(LAGS):] / self.scale[:, None],
                              np.zeros((y_hist.shape[0], H), dtype=np.float32)], axis=1)
        cal = _cal_cols(times_future)
        base = max(LAGS)
        for k in range(H):
            t = base + k
            lags = buf[:, [t - l for l in LAGS]]
            X = np.concatenate([lags, np.broadcast_to(cal[k], (len(buf), cal.shape[1]))], axis=1)
            buf[:, t] = self.booster.predict(X, num_threads=0)
        return (buf[:, base:] * self.scale[:, None]).astype(np.float32)
