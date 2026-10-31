# Lab 9 · Series de tiempo y forecasting (CC3092)

Pronóstico *day-ahead* (H = 24 h) del consumo horario de 315 clientes de ElectricityLoadDiagrams20112014.
Entrenamiento 2012-01 → 2014-06, validación 2014-07 → 2014-09, prueba 2014-10 → 2014-12 con backtesting de origen móvil diario (92 orígenes).

Este README dice qué está hecho (Parte 1) y qué falta (Parte 2), y cómo continuar sin repetir trabajo.

| | Responsable | Estado |
|---|---|---|
| Sección 2: exploración, preparación, `Dataset`, auditoría de fugas | Persona 1 | ✅ |
| Sección 4: `backtest()` y métricas | Persona 1 | ✅ (falta el reentrenamiento mensual) |
| Sección 5: baselines, LightGBM, MLP/DLinear/NLinear, LSTM, Seq2seq | Persona 1 | ✅ |
| Sección 5.1: estrategias multi-paso | Persona 1 | ✅ |
| Sección 5: N-HiTS/N-BEATS, TCN, PatchTST, TFT/iTransformer, DeepAR, Chronos-Bolt | **Persona 2** | ⬜ |
| Sección 5.2: iteraciones de hiperparámetros | **Persona 2** | ⬜ |
| Sección 5.3: modelo local contra global | **Persona 2** | ⬜ |
| Sección 4: reentrenamiento mensual | **Persona 2** | ⬜ |
| Sección 6: tabla comparativa y 3 gráficas | **Persona 2** | ⬜ |
| Sección 8: discusión (incluye 3 semillas) | **Persona 2** | ⬜ |
| Sección 3: investigación | Ambos | ⬜ |
| PDF (máx. 5 páginas) con el enlace al repo | Ambos | ⬜ |

> El enunciado salta de la sección 6 a la 8; no existe una sección 7.

---

## 1. Lo que ya está (Parte 1)

Todo vive en [`parte1_datos_y_modelos.ipynb`](parte1_datos_y_modelos.ipynb), ya ejecutado completo en una RTX 4050 (~1.5 h). Las celdas de análisis ya tienen escritas las respuestas con los números reales.

### Resultados de prueba (H = 24, 92 orígenes)

| Modelo | Familia | MASE | MAE (kWh) | MASE 22–28 dic | Entrenamiento |
|---|---|---|---|---|---|
| Seq2Seq | Encoder–decoder | **0.831** | 43.14 | 1.414 | 1,223 s |
| LSTM_AR | Recurrente (autorregresiva) | 0.832 | 43.30 | 1.394 | 1,292 s |
| MLP | MLP | 0.836 | 43.35 | 1.449 | 46 s |
| LSTM | Recurrente (multi-salida) | 0.871 | 44.64 | 1.566 | 914 s |
| LightGBM | Árboles | 0.880 | 43.85 | 1.717 | 47 s |
| NLinear | Lineal | 0.882 | 45.44 | 1.829 | 32 s |
| DLinear | Lineal | 0.885 | 45.52 | 1.837 | 22 s |
| MSTL | Baseline | 0.926 | 48.63 | 1.964 | (554 s de inferencia) |
| SeasonalNaive24 | Baseline | 1.044 | 50.01 | 2.235 | — |
| SeasonalNaive168 | Baseline | 1.074 | 56.63 | 1.623 | — |
| Naive | Baseline | 3.985 | 215.58 | 4.078 | — |

**Decisiones ya tomadas que la Parte 2 debe respetar:**

- **Panel:** 315 clientes (filtro: consumo desde 2012-01-01 y ≤ 7 días completos en cero). Son 26,304 horas por serie.
- **Cambio de horario:**
  - marzo: la hora faltante (01:00) se interpola;
  - octubre: la hora 01:00 se divide entre 2.
- **Normalización:** *por ventana* (tipo RevIN). Ganó en validación con MASE 0.877, contra 0.883 por serie y 3.23 global.
- **Calendario:** hora, día de la semana y mes con seno/coseno, más un binario de feriado de Portugal. Son 7 columnas, en `lab9.data.CAL_FEATURES`.
- **MASE:** la escala es el MAE in-sample del seasonal naive m = 24, calculado solo con entrenamiento. Está en `data/series_stats.csv`.
- **Protocolo:** todo modelo se evalúa con la misma función `backtest()`:
  - en cada origen solo recibe datos hasta las 23:00 del día anterior;
  - los pesos se entrenan una vez con datos hasta 2014-09-30;
  - la validación (jul–sep) se usa solo para early stopping y selección.
- **En este protocolo h coincide con la hora del día** (todos los orígenes son a las 00:00). Por eso el error por paso de horizonte tiene la forma del ciclo diario.

### Archivos que puedes usar directamente

| Archivo | Contenido |
|---|---|
| `data/hourly.parquet` | Panel limpio en formato largo (`unique_id, ds, y`), 8.3 M filas |
| `data/series_stats.csv` | `mean`, `std` y `mase_scale` por cliente, solo con entrenamiento |
| `data/clients.csv`, `data/holidays_pt.csv` | Clientes seleccionados y feriados usados |
| `preds/<modelo>.parquet` | Predicciones de prueba de cada modelo (`unique_id, ds, cutoff, h, y, y_hat, model`) |
| `preds/test_predictions_parte1.parquet` | Todos los modelos de la Parte 1 en el formato del entregable (`unique_id, ds, modelo, y_hat`) |
| `results/test_metrics_parte1.csv` | Tabla comparativa de la Parte 1 (las columnas que pide la sección 6) |
| `results/test_by_horizon.csv`, `test_by_hour.csv`, `test_by_client.csv` | Desgloses por paso, hora y cliente |
| `results/runs.csv` + `results/history/<run_id>.csv` | Una fila por entrenamiento con su configuración, parámetros, tiempos, memoria GPU y MAE/MASE de validación, más la curva por epoch. Ya incluye 3 iteraciones de normalización (`MLP_norm_*`). |
| `checkpoints/*.pt` | Pesos de las redes propias |
| `figures/*.png` | 12 gráficas de las secciones 2, 4 y 5 |

### El paquete `lab9/`

| Módulo | Para qué lo usarás |
|---|---|
| `config` | Fechas de las particiones, rutas, `H`, `SEED` |
| `data` | `load_panel()` → `Panel` (`Y` de forma (N, T), `times`, `ids`, `cal`), `calendar_features()`, `panel.until(ts)`, `panel.subset(ids)` |
| `backtest` | `backtest(model, panel, split="test" \| "val", H=24, refit_every=None \| "MS")` → `(preds, info)` |
| `metrics` | `evaluate(df, stats, name)`, `by(df, stats, "h" \| "hour" \| "unique_id")`. Si el DataFrame trae columnas `q0.1`, `q0.5`, `q0.9`, calcula también `pinball` y `cov_P10_P90`. |
| `baselines` | `StatsForecastModel`, `LightGBMGlobal(history_months=...)` |
| `models` | Redes propias: `build_mlp`, `build_dlinear`, `build_nlinear`, `build_rnn_mo`, `build_rnn_ar`, `build_seq2seq` |
| `training` | `TorchForecaster(...)` (fit/predict con early stopping y registro completo) y `save_run(model, info)` |

> ⚠️ **Los archivos de `lab9/` los genera el notebook de la Parte 1 con `%%writefile`.**
> - Si editas uno a mano y alguien vuelve a correr ese notebook, tu cambio se pierde.
> - El código nuevo va en un módulo nuevo, por ejemplo `lab9/neural.py`, escrito con `%%writefile` desde el notebook de la Parte 2.
> - Si de verdad hay que cambiar un módulo existente, cámbialo también en su celda del notebook 1.

---

## 2. Cómo correrlo

```powershell
pip install -r requirements.txt
pip install neuralforecast chronos-forecasting
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- Después de instalar `neuralforecast` y `chronos-forecasting`, revisa que `torch` siga siendo la build con CUDA (`2.6.0+cu124`, `True`). Si pip la reemplazó por la de CPU, reinstálala con:
  ```powershell
  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  ```
- Usamos pandas 2.3.3 porque `statsforecast` 2.1 no soporta pandas 3. No lo actualices.
- **No vuelvas a ejecutar el notebook de la Parte 1.** Tarda ~1.5 h y sus resultados ya están en `results/` y `preds/`. Además, las respuestas escritas citan esos números exactos. Si alguna vez hay que correrlo, `LAB9_QUICK=1` hace una corrida corta solo para comprobar que no truene. Sus números no valen.
- Crea un notebook nuevo, `parte2_modelos_y_analisis.ipynb`, en la raíz del repo. Empieza así:

```python
import pandas as pd, numpy as np, torch
from functools import partial
from lab9 import config as C, data as D, backtest as BT, metrics as ME, models as MD, training as TR
from lab9.baselines import LightGBMGlobal, StatsForecastModel

panel = D.load_panel()                         # los mismos 315 clientes
stats = pd.read_csv(C.STATS_PATH)              # escala del MASE
res1  = pd.read_csv(C.RESULTS_DIR / "test_metrics_parte1.csv")
```

---

## 3. Lo que falta (Parte 2)

### 3.1 Modelos de librería (filas restantes de la tabla de la sección 5)

| Familia | Modelos | Librería |
|---|---|---|
| Residuales y convolucionales | N-HiTS **o** N-BEATS, **y** TCN | NeuralForecast |
| Atención | PatchTST **y** uno entre TFT e iTransformer | NeuralForecast |
| Probabilísticos | DeepAR (con cuantiles P10, P50, P90) | NeuralForecast o GluonTS |
| Fundacional | Chronos-Bolt *zero-shot* (sin entrenamiento, solo inferencia) | chronos-forecasting |

**Requisito:** que pasen por `BT.backtest()` igual que los demás, para que las métricas y los tiempos sean comparables. Solo necesitan una clase con este contrato:

```python
model.name                                         # str
model.fit(panel, val_start)                        # panel ya truncado a 2014-09-30; val_start = 2014-07-01
model.predict(y_hist, times_hist, times_future)    # y_hist (N, t) en kWh, solo el pasado
    # -> ndarray (N, H)  o  {"mean": (N, H), "q0.1": (N, H), "q0.5": (N, H), "q0.9": (N, H)}
```

Si `predict` devuelve el diccionario con cuantiles, `backtest` los agrega como columnas y `ME.evaluate` calcula solo la pinball loss y la cobertura P10–P90.

**Esbozo para NeuralForecast** (no está probado; ajústalo a la versión que instales):

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TCN, PatchTST, TFT, DeepAR
from neuralforecast.losses.pytorch import MAE, DistributionLoss

class NFModel:
    """Envuelve un modelo de NeuralForecast con el contrato fit/predict de backtest()."""
    def __init__(self, name, nf_model, L, exog=True, history_months=None):
        self.name, self.m, self.L, self.exog, self.history_months = name, nf_model, L, exog, history_months

    def _long(self, Y, times, ids):
        df = pd.DataFrame({"unique_id": np.repeat(ids, len(times)), "ds": np.tile(times, len(ids)),
                           "y": Y.reshape(-1)})
        if self.exog:
            cal = D.calendar_features(times).reset_index(drop=True)
            df = pd.concat([df, pd.concat([cal] * len(ids), ignore_index=True)], axis=1)
        return df

    def fit(self, panel, val_start):
        start = 0
        if self.history_months:      # historia de entrenamiento limitada (sección 5.2)
            first = pd.Timestamp(val_start) - pd.DateOffset(months=self.history_months)
            start = max(0, panel.idx(first) - self.L)
        self.ids = panel.ids
        df = self._long(panel.Y[:, start:], panel.times[start:], panel.ids)
        self.nf = NeuralForecast(models=[self.m], freq="h")
        self.nf.fit(df=df, val_size=len(panel.times) - panel.idx(val_start))   # validación = jul–sep
        return self

    def predict(self, y_hist, times_hist, times_future):
        hist = self._long(y_hist[:, -self.L:], times_hist[-self.L:], self.ids)
        futr = self._long(np.zeros((len(self.ids), len(times_future))), times_future, self.ids).drop(columns="y")
        fc = self.nf.predict(df=hist, futr_df=futr if self.exog else None)
        fc = fc.set_index(["unique_id", "ds"]).loc[pd.MultiIndex.from_product([self.ids, times_future])]
        pick = lambda col: fc[col].to_numpy().reshape(len(self.ids), len(times_future))
        out = {"mean": pick(self.name)}
        if f"{self.name}-lo-80" in fc:          # DeepAR: el intervalo de 80% equivale a P10–P90
            out |= {"q0.1": pick(f"{self.name}-lo-80"), "q0.5": pick(f"{self.name}-median"),
                    "q0.9": pick(f"{self.name}-hi-80")}
        return out

cal = list(D.CAL_FEATURES)
common = dict(h=24, input_size=168, max_steps=2000, early_stop_patience_steps=5, val_check_steps=100,
              scaler_type="robust", random_seed=C.SEED)
nhits = NFModel("NHITS", NHITS(futr_exog_list=cal, **common), L=168)
preds, info = BT.backtest(nhits, panel, split="test")
```

Notas para cada modelo:

- **El nombre del wrapper debe coincidir con el `alias` del modelo.** Por defecto la columna de NeuralForecast se llama como la clase (`NHITS`, `PatchTST`…), así que el `name` del wrapper tiene que ser igual, o se pasa `alias=` al modelo.
- **PatchTST e iTransformer no aceptan covariables futuras:** usa `exog=False`. iTransformer además es multivariado y necesita `n_series=315`.
- **Para DeepAR:** `loss=DistributionLoss("StudentT", level=[80])`.
- **Para Chronos-Bolt** basta un wrapper con `fit` vacío. En `predict` se llama a `pipeline.predict_quantiles` con `y_hist[:, -2048:]` (el contexto máximo de Bolt), `prediction_length=len(times_future)` y `quantile_levels=[0.1, 0.5, 0.9]`, y se devuelve el diccionario con `q0.1`, `q0.5` y `q0.9`. Cárgalo con:
  ```python
  BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cuda", torch_dtype=torch.bfloat16)
  ```
- **Por cada modelo guarda:**
  - `preds.to_parquet(C.PREDS_DIR / f"{name}.parquet")`;
  - una fila con `ME.evaluate(preds, stats, name, extra={...})`, con las mismas columnas extra que `test_metrics_parte1.csv`: `family`, `L`, `strategy`, `uses_cov`, `params_total`, `params_trainable`, `train_time_s`, `infer_time_s`, `hardware`, `gpu_peak_mb`;
  - los parámetros, con `sum(p.numel() for p in nf.models[0].parameters())`;
  - la memoria pico, con `torch.cuda.max_memory_allocated()` (llama antes a `torch.cuda.reset_peak_memory_stats()`).

### 3.2 Iteraciones de hiperparámetros (5.2)

- **Obligatorio:**
  - una iteración que varíe **L** (48, 168, 336);
  - otra que varíe los **meses de historia** (3, 12, 30).
- **Otros hiperparámetros:** normalización (ya hecha, 3 corridas), tamaño del estado oculto, capas, dropout, learning rate y batch.
- **Solo validación:** usa `split="val"`. La prueba se corre **una sola vez** por modelo, con la configuración elegida.
- **Las redes propias ya registran todo** lo que pide el enunciado:

```python
m = TR.TorchForecaster("Seq2Seq_L336", partial(MD.build_seq2seq, hidden=128, n_layers=2),
                       L=336, history_months=12, norm="window", family="Encoder–decoder",
                       strategy="encoder–decoder")
preds_val, info = BT.backtest(m, panel, split="val")   # entrena, y mide la inferencia en los 92 orígenes de validación
TR.save_run(m, info)    # -> results/runs.csv + results/history/<run_id>.csv (pérdidas, MAE y MASE por epoch)
```

  `runs.csv` queda con la configuración (L, H, meses de historia…), los parámetros totales y entrenables, el tiempo por epoch y total, el tiempo de inferencia y la memoria pico de GPU.
- **`LightGBMGlobal(history_months=M)`** también acepta meses de historia. Para NeuralForecast usa `history_months` del esbozo de arriba.
- **Gráfica:** las curvas de pérdida de al menos 3 iteraciones, desde `results/history/*.csv`. Responde: *¿una pérdida de entrenamiento más baja implica siempre un mejor MASE en validación?*
  - Pista: `LSTM_AR` tiene una pérdida de entrenamiento de 0.17 (por el *teacher forcing*) contra 0.26 en validación, y no le gana al Seq2Seq.
- **Presupuesto en la RTX 4050:**

  | Modelo | Tiempo por corrida | Tiempo por epoch |
  |---|---|---|
  | MLP / DLinear | ~1 min | — |
  | LSTM / Seq2Seq (L = 168) | 15–20 min | ~25–45 s |
  | LSTM / Seq2Seq (L = 336) | aproximadamente el doble | — |

  Con 30 epochs ninguna LSTM llegó al early stopping. Puede valer la pena subir `max_epochs`.
- **Corridas sugeridas:**
  - L ∈ {48, 168, 336} y meses ∈ {3, 12, 30} con el mejor recurrente;
  - meses ∈ {3, 12, 30} con LightGBM, DLinear y PatchTST, porque los pide la gráfica 3 de la sección 6;
  - 2–3 corridas más de hidden, lr o dropout.

### 3.3 Modelo local contra global (5.3)

- **Modelo:** el mejor modelo propio. Es Seq2Seq (MASE 0.831); el MLP queda casi empatado (0.836) y es 26 veces más barato, así que también sirve.
- **Clientes:** 10, elegidos con semilla fija. Conviene que cubran escalas distintas (`stats["mean"]`).
- **Modelo local:** un `TorchForecaster` por cliente, con `BT.backtest(m, panel.subset([cid]), split="test")`.
- **Modelo global:** ya está; filtra `preds/Seq2Seq.parquet` a esos 10 clientes.
- **Comparación:** MASE por cliente con `ME.by(df, stats, "unique_id")`. *¿Qué clientes se benefician más del global?*

### 3.4 Reentrenamiento mensual (sección 4)

`backtest` ya lo soporta:

```python
preds, info = BT.backtest(model, panel, split="test", refit_every="MS")
```

- **Reentrena** el 1 de octubre, el 1 de noviembre y el 1 de diciembre, con ventana expandible y los 3 meses previos como validación.
- **Se pide** para el mejor modelo de deep learning y para el mejor baseline. Compara su MASE con la versión entrenada una sola vez.
- **Ojo con el baseline:** el mejor es MSTL, que es local y ya se reajusta en cada origen. Su resultado con reentrenamiento mensual es idéntico por construcción; explícalo así en el reporte. Si quieren un caso con pesos que sí cambian, agreguen LightGBM.

### 3.5 Tabla comparativa y gráficas (sección 6)

**Tabla:** une `results/test_metrics_parte1.csv` con las filas de la Parte 2, ordenada por MASE. Tiene que incluir:

- familia, L, estrategia multi-paso, si usa covariables;
- parámetros totales y entrenables;
- tiempo de entrenamiento, tiempo de inferencia y hardware (Chronos: solo inferencia);
- MAE, RMSE, WAPE y MASE con H = 24, más el **MASE en la semana del 22 al 28 de diciembre** (`MASE_xmas`, ya lo calcula `evaluate`);
- **MASE con H = 168** "cuando aplique". No está hecho en la Parte 1.
  - Se obtiene con `BT.backtest(m, panel, H=168)`: 86 orígenes, los días cuyo horizonte de 168 h cabe en la prueba.
  - Las redes propias se crean con `TorchForecaster(..., H=168)`.
  - StatsForecast y LightGBM toman el horizonte de `times_future`, así que funcionan sin cambios.
  - Hazlo al menos para los baselines, LightGBM, el mejor propio y alguno de NeuralForecast.
- **pinball** y **cov_P10_P90** para DeepAR y Chronos.

**Gráficas** (como mínimo 3):

1. **MASE de prueba contra tiempo de entrenamiento** (eje x en log), un punto por modelo. Marca la frontera de Pareto.
2. **MAE por paso del horizonte** (h = 1…24), una curva por modelo.
   - Las de la Parte 1 están en `results/test_by_horizon.csv`.
   - Con más de 8 modelos, divídela en dos paneles para que se lea.
3. **MASE contra meses de historia** (3, 12, 30) para seasonal naive, LightGBM, DLinear, el mejor recurrente y PatchTST, con Chronos como línea horizontal de referencia.
   - Usa el MASE de validación de las corridas de 5.2.
   - El seasonal naive no depende de la historia: también es una línea plana.

**Archivo del entregable:** `preds/test_predictions.parquet` con columnas `unique_id, ds, modelo, y_hat`. Contiene **todos** los modelos: concatena `test_predictions_parte1.parquet` con los de la Parte 2.

Para las gráficas se usó una paleta fija, en este orden: `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #6250d6 #e34948`; gris `#898781`, fondo `#fcfcfb`. Usarla da consistencia con las figuras de la Parte 1.

### 3.6 Discusión (sección 8)

Respuestas que hay que escribir con base en los resultados, y lo que ya hay de la Parte 1 para cada una:

- **¿Qué modelo ganó y por cuánto superó al seasonal naive? ¿Vale su costo?**
  - Hoy Seq2Seq le gana al SN24 por 20% de MASE (0.831 contra 1.044).
  - El MLP consigue casi lo mismo en 46 s.
- **¿DLinear compite con los Transformers? ¿En qué horizonte?** Compara DLinear con PatchTST, con H = 24 y con H = 168.
- **¿Qué estrategia multi-paso funcionó mejor? ¿Cómo crece el error por familia?**
  - La 5.1 ya está respondida en el notebook 1: encoder–decoder ≈ autorregresiva > multi-salida.
  - Falta agregar las familias nuevas.
- **¿Qué efecto tuvieron L, la normalización y la historia? ¿A partir de cuántos meses el deep learning supera a LightGBM y a los estadísticos?** Sale de 5.2 y de la gráfica 3.
- **¿Global o por cliente? ¿Quién se beneficia del global?** Sale de 5.3.
- **¿Qué tan cerca quedó Chronos zero-shot? ¿Cuándo lo usarían?**
- **¿Qué pasó en Navidad y Año Nuevo? ¿Ayudó la covariable de feriado?**
  - Ya hay análisis en el notebook 1: todos empeoran entre 68% y 114%, y el 25 de diciembre concentra el error.
  - Falta ver si los modelos nuevos se comportan igual.
- **Sensibilidad a la semilla:** repite el mejor modelo con 3 semillas y reporta la desviación estándar del MASE.
  - Redes propias: `TorchForecaster(..., seed=s)`.
  - NeuralForecast: `random_seed=s`.

### 3.7 Entre los dos

- **Sección 3 (investigación):** `nn.LSTM`/`nn.GRU`, `nn.Conv1d` con dilatación y campo receptivo de la TCN, `nn.TransformerEncoderLayer` y codificación posicional, `cross_validation` de Nixtla (`n_windows`, `step_size`, `refit`), `predict_quantiles` de Chronos. Conceptos: local contra global, teacher forcing, RevIN, pinball loss, MASE contra WAPE.
- **PDF de máximo 5 páginas:** investigación, preparación y auditoría de fugas, tabla de iteraciones, estrategias multi-paso, tabla comparativa, resultados probabilísticos, gráficas, análisis y conclusiones. Al final va el **enlace al repositorio**.
- **Repositorio:** los notebooks comentados más `preds/test_predictions.parquet`. Los datos crudos de UCI no se suben (`data/raw/` está ignorado). `data/hourly.parquet` pesa 47 MB, por debajo del límite de 100 MB de GitHub.

---

## Checklist de la Persona 2

- [ ] Instalar `neuralforecast` y `chronos-forecasting` y verificar que torch siga con CUDA
- [ ] Wrapper de NeuralForecast + N-HiTS (o N-BEATS) y TCN
- [ ] PatchTST + TFT o iTransformer
- [ ] DeepAR con P10/P50/P90 → pinball y cobertura
- [ ] Chronos-Bolt zero-shot → pinball y cobertura
- [ ] 5.2: iteraciones de L (48/168/336) y de meses de historia (3/12/30), más otros hiperparámetros, solo en validación
- [ ] 5.2: gráfica de curvas de pérdida (≥ 3 iteraciones)
- [ ] 5.3: 10 modelos locales contra el global
- [ ] Reentrenamiento mensual del mejor DL y del mejor baseline
- [ ] MASE con H = 168 donde aplique
- [ ] Sección 6: tabla final ordenada por MASE + 3 gráficas
- [ ] 3 semillas del mejor modelo → desviación estándar del MASE
- [ ] `preds/test_predictions.parquet` con todos los modelos
- [ ] Sección 8: discusión
