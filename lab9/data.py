"""Etapas 1-2 del pipeline: datos crudos (15 min, kW) -> panel horario limpio (kWh)."""
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Descarga y lectura
# ---------------------------------------------------------------------------
def download_raw():
    """Descarga el zip de UCI solo si el .txt no existe todavía."""
    C.ensure_dirs()
    if not C.RAW_TXT.exists():
        zip_path = C.RAW_DIR / "electricity.zip"
        urllib.request.urlretrieve(C.URL, zip_path)
        zipfile.ZipFile(zip_path).extract("LD2011_2014.txt", C.RAW_DIR)
    return C.RAW_TXT


def load_raw():
    return pd.read_csv(C.RAW_TXT, sep=";", decimal=",", index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Filtro de clientes
# ---------------------------------------------------------------------------
def select_clients(raw, max_zero_days=7):
    """Clientes con datos completos en 2012-2014.

    En el archivo, un cliente que todavía no existe aparece con ceros. Un cliente es válido si
    (1) ya consume en la primera hora de 2012-01-01 y (2) no tiene más de `max_zero_days`
    días completos en cero dentro del periodo (bajas o medidores apagados).
    Devuelve la lista de clientes y una tabla con el diagnóstico de cada uno.
    """
    period = raw.loc[C.START:C.END + pd.Timedelta(minutes=45)]
    first_hour = period.loc[C.START:C.START + pd.Timedelta(minutes=45)]
    starts_ok = first_hour.ne(0).any()
    daily = period.resample("D").sum()
    zero_days = daily.eq(0).sum()
    diag = pd.DataFrame({"starts_2012": starts_ok, "zero_days": zero_days})
    diag["keep"] = diag["starts_2012"] & (diag["zero_days"] <= max_zero_days)
    clients = diag.index[diag["keep"]].tolist()
    if C.N_CLIENTS is not None:
        rng = np.random.default_rng(C.SEED)
        clients = sorted(rng.choice(clients, size=C.N_CLIENTS, replace=False).tolist())
    return clients, diag


# ---------------------------------------------------------------------------
# 15 min -> horas, y días de cambio de horario
# ---------------------------------------------------------------------------
def dst_days(years=(2012, 2013, 2014)):
    """Último domingo de marzo (23 h) y de octubre (25 h) de cada año."""
    def last_sunday(y, m):
        d = pd.Timestamp(year=y, month=m, day=31)
        return d - pd.Timedelta(days=(d.weekday() - 6) % 7)

    return {"spring": [last_sunday(y, 3) for y in years],
            "autumn": [last_sunday(y, 10) for y in years]}


def to_hourly(raw, clients):
    """kW cada 15 min -> kWh por hora.

    Cada registro es la potencia media (kW) de 15 minutos, así que la energía de la hora es
    sum(kW) / 4. La hora h agrupa las etiquetas h:00, h:15, h:30 y h:45 (closed='left'): con esa
    convención las anomalías de cambio de horario quedan contenidas en una sola hora (01:00).
    """
    sub = raw.loc[C.START:C.END + pd.Timedelta(minutes=45), clients]
    hourly = sub.resample("h", closed="left", label="left").sum() / 4.0
    assert hourly.index[0] == C.START and hourly.index[-1] == C.END
    return hourly


def fix_dst(hourly):
    """Corrige las dos anomalías anuales de cambio de horario (documentadas por UCI).

    - Marzo (día de 23 h): la hora 01:00 no existe y viene en cero -> se imputa con la media de
      las horas 00:00 y 02:00 del mismo día (interpolación lineal, solo usa vecinos inmediatos).
    - Octubre (día de 25 h): la hora 01:00 acumula dos horas reales -> se divide entre 2.
    Así el panel mantiene una rejilla regular de 24 h/día en hora local, que es la que siguen
    los hábitos de consumo y las covariables de calendario.
    """
    out = hourly.copy()
    days = dst_days()
    for d in days["spring"]:
        t = d + pd.Timedelta(hours=1)
        out.loc[t] = 0.5 * (out.loc[t - pd.Timedelta(hours=1)] + out.loc[t + pd.Timedelta(hours=1)])
    for d in days["autumn"]:
        t = d + pd.Timedelta(hours=1)
        out.loc[t] = out.loc[t] / 2.0
    return out


def to_long(wide):
    """Matriz (tiempo x cliente) -> formato largo de Nixtla (unique_id, ds, y)."""
    long = wide.rename_axis("ds").reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    return long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def from_long(long):
    return long.pivot(index="ds", columns="unique_id", values="y").sort_index()


# ---------------------------------------------------------------------------
# Feriados y covariables de calendario (conocidas a futuro)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def portugal_holidays(years=(2012, 2013, 2014)):
    """Feriados nacionales de Portugal. Entre 2013 y 2015 se suspendieron Corpus Christi,
    5 de octubre, 1 de noviembre y 1 de diciembre; la librería `holidays` ya lo refleja."""
    import holidays
    hol = holidays.Portugal(years=list(years))
    return pd.Series({pd.Timestamp(k): v for k, v in sorted(hol.items())}, name="holiday")


CAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "holiday"]


def calendar_features(index, holiday_dates=None):
    """Covariables de calendario para cada marca horaria.

    Codificación seno/coseno para las variables cíclicas: 2 columnas por variable (en lugar de
    24+7+12 con one-hot) y conservan la cercanía circular (23:00 queda junto a 00:00, diciembre
    junto a enero). El feriado es binario. Todas dependen solo de la fecha, así que se conocen
    con certeza para cualquier hora futura.
    """
    index = pd.DatetimeIndex(index)
    if holiday_dates is None:
        holiday_dates = portugal_holidays().index
    hol_days = pd.DatetimeIndex(holiday_dates).normalize()
    hour = index.hour.values
    dow = index.dayofweek.values
    month = index.month.values - 1
    feats = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        index.normalize().isin(hol_days).astype(float),
    ], axis=1).astype(np.float32)
    return pd.DataFrame(feats, index=index, columns=CAL_FEATURES)


# ---------------------------------------------------------------------------
# Panel: contenedor común que reciben todos los modelos
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    """Panel horario en forma de matriz.

    Y: (N, T) en kWh; times: DatetimeIndex de longitud T; ids: N nombres de cliente;
    cal: (T, n_cal) covariables de calendario alineadas con `times`.
    """
    Y: np.ndarray
    times: pd.DatetimeIndex
    ids: list
    cal: np.ndarray

    @classmethod
    def from_wide(cls, wide):
        cal = calendar_features(wide.index).values
        return cls(wide.T.values.astype(np.float32), pd.DatetimeIndex(wide.index), list(wide.columns), cal)

    @property
    def N(self):
        return self.Y.shape[0]

    def idx(self, ts):
        """Posición entera de la marca `ts` en el eje temporal."""
        return int(self.times.get_loc(pd.Timestamp(ts)))

    def until(self, ts):
        """Panel truncado: solo observaciones con marca <= ts. Es lo único que ve fit()."""
        end = self.idx(ts) + 1
        return Panel(self.Y[:, :end], self.times[:end], self.ids, self.cal[:end])

    def subset(self, ids):
        pos = [self.ids.index(i) for i in ids]
        return Panel(self.Y[pos], self.times, list(ids), self.cal)

    def to_wide(self):
        return pd.DataFrame(self.Y.T, index=self.times, columns=self.ids)


def load_panel():
    """Lee el panel limpio exportado por el notebook (data/hourly.parquet)."""
    long = pd.read_parquet(C.HOURLY_PATH)
    return Panel.from_wide(from_long(long))


def series_stats(panel, train_end=C.TRAIN_END, season=C.SEASON):
    """Estadísticos por serie calculados SOLO con entrenamiento.

    mean/std -> z-score por serie; mase_scale -> MAE in-sample del seasonal naive (m=24).
    """
    tr = panel.until(train_end).Y
    mase_scale = np.abs(tr[:, season:] - tr[:, :-season]).mean(axis=1)
    return pd.DataFrame({
        "unique_id": panel.ids,
        "mean": tr.mean(axis=1),
        "std": tr.std(axis=1),
        "mase_scale": mase_scale,
    })
