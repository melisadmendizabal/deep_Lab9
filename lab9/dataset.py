"""Etapa 3 del pipeline: series horarias -> ventanas deslizantes -> tensores."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C

NORMS = ("series", "global", "window", "none")


def norm_params(panel_train, norm):
    """Parámetros de normalización estimados SOLO con el panel de entrenamiento recibido.

    Devuelve (loc, scale) por serie, de forma (N,). Para 'window' la media/desviación se
    calculan en cada ventana de entrada; aquí solo se devuelve la desviación de
    entrenamiento, que se usa como piso para no dividir entre ~0.
    """
    Y = panel_train.Y
    if norm == "series":
        return Y.mean(axis=1), Y.std(axis=1) + 1e-6
    if norm == "global":
        return np.full(len(Y), Y.mean()), np.full(len(Y), Y.std())
    if norm == "window":
        return np.zeros(len(Y)), 0.1 * Y.std(axis=1) + 1e-6
    if norm == "none":
        return np.zeros(len(Y)), np.ones(len(Y))
    raise ValueError(f"norm debe ser uno de {NORMS}")


class WindowDataset(Dataset):
    """Ventanas deslizantes sobre TODAS las series del panel.

    Cada muestra es (serie i, inicio del objetivo s):
      x     : (L, 1 + n_cal)  [y normalizada, calendario] en las horas s-L ... s-1
      x_fut : (H, n_cal)      calendario de las horas objetivo s ... s+H-1 (conocido a futuro)
      y     : (H,)            y normalizada en s ... s+H-1
      loc, scale : escalares para regresar a kWh (y_kwh = y * scale + loc)
      sid   : índice de la serie
    Con un DataLoader de batch B se obtiene X: (B, L, features) e y: (B, H).

    Solo se generan ventanas cuyo objetivo cae completo dentro de [first_target, last_target];
    la entrada puede venir de antes (es pasado). `stride` es el paso entre orígenes
    consecutivos (1 = cada hora, 24 = un origen por día a las 00:00).
    `history_months` limita los datos usados (entrada + objetivo) a los últimos M meses.
    """

    def __init__(self, panel, L, H, first_target, last_target, loc, scale, norm="series",
                 stride=1, history_months=None):
        last_pos = panel.idx(last_target)
        # El panel se trunca en el último objetivo: es imposible leer datos posteriores.
        panel = panel.until(last_target)
        first_pos = panel.idx(first_target)
        min_input = 0
        if history_months is not None:
            data_start = pd.Timestamp(last_target) - pd.DateOffset(months=history_months) + pd.Timedelta(hours=1)
            min_input = max(0, int(np.searchsorted(panel.times, data_start)))
        s0 = max(first_pos, min_input + L)
        self.starts = np.arange(s0, last_pos - H + 2, stride)
        self.L, self.H, self.norm = L, H, norm
        self.times = panel.times
        self.Y = torch.as_tensor(panel.Y, dtype=torch.float32)               # (N, T)
        self.cal = torch.as_tensor(panel.cal, dtype=torch.float32)           # (T, n_cal)
        self.loc = torch.as_tensor(np.asarray(loc), dtype=torch.float32)
        self.scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
        self.n_series = self.Y.shape[0]
        self.n_per_series = len(self.starts)
        self.n_features = 1 + self.cal.shape[1]
        self._in = torch.arange(-L, 0)
        self._out = torch.arange(0, H)

    def __len__(self):
        return self.n_series * self.n_per_series

    def __getitem__(self, i):
        x, xf, y, loc, scale, sid = self.__getitems__([i])
        return x[0], xf[0], y[0], loc[0], scale[0], sid[0]

    def __getitems__(self, idx):
        """Versión vectorizada: el DataLoader la usa para armar el batch completo de una vez."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        sid = idx // self.n_per_series
        s = torch.as_tensor(self.starts)[idx % self.n_per_series]
        pin = s[:, None] + self._in                                          # (B, L)
        pout = s[:, None] + self._out                                        # (B, H)
        y_in = self.Y[sid[:, None], pin]
        y_out = self.Y[sid[:, None], pout]
        if self.norm == "window":
            # Estadísticos de la propia ventana de entrada (disponibles en el origen).
            loc = y_in.mean(dim=1)
            scale = torch.maximum(y_in.std(dim=1), self.scale[sid])
        else:
            loc, scale = self.loc[sid], self.scale[sid]
        y_in = (y_in - loc[:, None]) / scale[:, None]
        y_out = (y_out - loc[:, None]) / scale[:, None]
        x = torch.cat([y_in[..., None], self.cal[pin]], dim=-1)
        xf = self.cal[pout]
        return x, xf, y_out, loc, scale, sid

    def target_times(self, i):
        s = self.starts[i % self.n_per_series]
        return self.times[s:s + self.H]


def collate_batch(batch):
    """__getitems__ ya devuelve tensores apilados; el collate no tiene que hacer nada."""
    return batch


def make_inputs(y_hist, times_hist, times_future, L, loc, scale, norm, cal_fn):
    """Arma (x, x_fut, loc, scale) para inferencia en un origen, con la misma lógica que el Dataset.

    y_hist: (N, t) en kWh con t >= L, solo datos anteriores al origen.
    """
    y_in = torch.as_tensor(np.asarray(y_hist[:, -L:]), dtype=torch.float32)
    loc = torch.as_tensor(np.asarray(loc), dtype=torch.float32)
    scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
    if norm == "window":
        loc = y_in.mean(dim=1)
        scale = torch.maximum(y_in.std(dim=1), scale)
    y_in = (y_in - loc[:, None]) / scale[:, None]
    N = y_in.shape[0]
    cal_in = torch.as_tensor(cal_fn(times_hist[-L:]).values).expand(N, -1, -1)
    cal_out = torch.as_tensor(cal_fn(times_future).values).expand(N, -1, -1)
    x = torch.cat([y_in[..., None], cal_in], dim=-1)
    return x, cal_out.contiguous(), loc, scale


def count_windows(panel, L=168, H=C.H, stride=1):
    """Número de ventanas por partición con stride dado (objetivo completo dentro del bloque)."""
    rows = []
    for split, a, b in [("train", C.START, C.TRAIN_END), ("val", C.VAL_START, C.VAL_END),
                        ("test", C.TEST_START, C.TEST_END)]:
        ds = WindowDataset(panel, L, H, a, b, np.zeros(panel.N), np.ones(panel.N), "none", stride)
        rows.append({"split": split, "stride": stride, "por_serie": ds.n_per_series,
                     "panel": len(ds)})
    return pd.DataFrame(rows)
