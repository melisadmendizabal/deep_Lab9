"""Constantes compartidas del laboratorio (fechas de corte, rutas, horizonte)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREDS_DIR = ROOT / "preds"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

URL = ("https://archive.ics.uci.edu/static/public/321/"
       "electricityloaddiagrams20112014.zip")
RAW_TXT = RAW_DIR / "LD2011_2014.txt"
HOURLY_PATH = DATA_DIR / "hourly.parquet"     # formato largo: unique_id, ds, y
CLIENTS_PATH = DATA_DIR / "clients.csv"
STATS_PATH = DATA_DIR / "series_stats.csv"

# Periodo de estudio y particiones (todas las marcas son el inicio de la hora).
START = pd.Timestamp("2012-01-01 00:00")
END = pd.Timestamp("2014-12-31 23:00")
TRAIN_END = pd.Timestamp("2014-06-30 23:00")
VAL_START = pd.Timestamp("2014-07-01 00:00")
VAL_END = pd.Timestamp("2014-09-30 23:00")
TEST_START = pd.Timestamp("2014-10-01 00:00")
TEST_END = END

H = 24                 # horizonte day-ahead
SEASON = 24            # estacionalidad usada en la escala del MASE
XMAS_START = pd.Timestamp("2014-12-22 00:00")
XMAS_END = pd.Timestamp("2014-12-28 23:00")

SEED = 42
N_CLIENTS = None       # None = todos los clientes que pasan el filtro; si no, >= 100


def ensure_dirs():
    for d in (DATA_DIR, RAW_DIR, PREDS_DIR, RESULTS_DIR, FIG_DIR,
              RESULTS_DIR / "history"):
        d.mkdir(parents=True, exist_ok=True)
