# Lab 9 · Series de tiempo y forecasting (CC3092)

Pronóstico *day-ahead* (H = 24 h) del consumo horario de 315 clientes de ElectricityLoadDiagrams20112014.
Entrenamiento 2012-01 → 2014-06, validación 2014-07 → 2014-09, prueba 2014-10 → 2014-12 con backtesting de origen móvil diario (92 orígenes).

Este README dice qué está hecho (Parte 1 y Parte 2), dónde están los resultados y qué falta (investigación, discusión escrita y PDF).

| | Responsable | Estado |
|---|---|---|
| Sección 2: exploración, preparación, `Dataset`, auditoría de fugas | Persona 1 | ✅ |
| Sección 4: `backtest()` y métricas | Persona 1 | ✅ |
| Sección 5: baselines, LightGBM, MLP/DLinear/NLinear, LSTM, Seq2seq | Persona 1 | ✅ |
| Sección 5.1: estrategias multi-paso | Persona 1 | ✅ |
| Sección 5: N-HiTS, TCN, PatchTST, TFT, DeepAR, Chronos-Bolt | Persona 2 | ✅ |
| Sección 5.2: iteraciones de hiperparámetros | Persona 2 | ✅ |
| Sección 5.3: modelo local contra global | Persona 2 | ✅ |
| Sección 4: reentrenamiento mensual | Persona 2 | ✅ |
| Sección 6: MASE con H = 168, tabla comparativa y 3 gráficas | Persona 2 | ✅ |
| Sección 8: 3 semillas y números de apoyo | Persona 2 | ✅ |
| Sección 8: respuestas escritas | Ambos | ⬜ (los números están en el notebook 2) |
| Sección 3: investigación | Ambos | ⬜ |
| PDF (máx. 5 páginas) con el enlace al repo | Ambos | ⬜ |

> El enunciado salta de la sección 6 a la 8; no existe una sección 7.

---

## 1. Parte 1

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
| `neural` (Parte 2) | `NFModel` (NeuralForecast) y `ChronosBolt`, con el mismo contrato fit/predict |

> ⚠️ **Los archivos de `lab9/` los generan los notebooks con `%%writefile`.**
> - `neural.py` lo escribe el notebook 2; los demás, el notebook 1.
> - Si editas uno a mano y alguien vuelve a correr su notebook, tu cambio se pierde. Cámbialo también en su celda.

---

## 2. Cómo correrlo

La Parte 2 se corrió en una RTX 4060 Laptop (8 GB) con un entorno virtual que reutiliza el torch con CUDA ya instalado:

```powershell
py -3.11 -m venv --system-site-packages .venv          # hereda torch 2.13.0+cu126 del Python del sistema
.venv\Scripts\python -m pip install statsforecast lightgbm neuralforecast chronos-forecasting holidays ipykernel
.venv\Scripts\python -m ipykernel install --user --name lab9 --display-name "Python (lab9 .venv)"
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- **Versiones de la Parte 2:**

  | Paquete | Versión |
  |---|---|
  | `neuralforecast` | 3.3.0 |
  | `chronos-forecasting` | 2.3.2 |
  | `statsforecast` | 2.1.1 |
  | `lightgbm` | 4.7.0 |
  | `pandas` | 2.2.2 |
  | `numpy` | 1.26.4 |
  | `torch` | 2.13.0+cu126 |

  `requirements.txt` sigue con los pines de la Parte 1 y ahora agrega `neuralforecast` y `chronos-forecasting`.
- **Torch tras instalar NeuralForecast y Chronos:** revisa que siga siendo la build con CUDA. Si pip la cambió por la de CPU, reinstala la build con CUDA de tu versión.
- **No vuelvas a ejecutar el notebook de la Parte 1.** Tarda ~1.5 h, sus resultados ya están en `results/` y `preds/`, y sus respuestas escritas citan esos números exactos. `LAB9_QUICK=1` hace una corrida corta solo para comprobar que no truene; sus números no valen.
- **El notebook de la Parte 2 tiene caché.** Cada corrida guarda su fila en `results/*.csv` y sus pronósticos en `preds/` al terminar. Volver a ejecutarlo completo carga lo ya calculado y solo corre lo que falta; tarda ~2 min.
  - `RECOMPUTE = True` (celda de configuración) fuerza a recalcular todo, unas 3–4 h en la RTX 4060.
  - `LAB9_QUICK=1` corre con 20 clientes y muy pocas epochs, y escribe en `quick/` (ignorado por git).

---

## 3. Parte 2

Todo vive en [`parte2_modelos_y_analisis.ipynb`](parte2_modelos_y_analisis.ipynb), ya ejecutado completo. Usa el mismo panel, la misma escala del MASE y la misma función `BT.backtest()` que la Parte 1.

> **Hardware distinto.** La Parte 1 se entrenó en una RTX 4050 y la Parte 2 en una RTX 4060 (laptop, ambas de 6–8 GB). Son GPUs de la misma generación y los tiempos son comparables en orden de magnitud, pero no al segundo. La columna `hardware` de la tabla final dice dónde corrió cada modelo.

### 3.1 Tabla comparativa final (prueba, H = 24, 92 orígenes)

Ordenada por MASE; está completa en `results/test_metrics.csv`. Aquí solo va la configuración elegida de cada modelo (columna `elegida`; ver 3.3).

| Modelo | Familia | MASE | MAE (kWh) | MASE 22–28 dic | MASE H = 168 | Entrenamiento |
|---|---|---|---|---|---|---|
| ChronosBolt (zero-shot) | Fundacional | **0.767** | 40.13 | 1.490 | **0.919** | — (31 s de inferencia) |
| NHITS | Residuales y convolucionales | 0.779 | 40.83 | **1.351** | 0.950 | 77 s |
| MLP_lr3e-4 | MLP y lineales | 0.812 | 41.75 | 1.437 | 0.975 | 47 s |
| Seq2Seq | Encoder–decoder | 0.831 | 43.14 | 1.414 | 0.998 | 1,223 s |
| LSTM_AR | Recurrentes | 0.832 | 43.30 | 1.394 | — | 1,292 s |
| TFT | Atención | 0.833 | 42.83 | 1.434 | — | 287 s |
| LSTM | Recurrentes | 0.871 | 44.64 | 1.566 | — | 914 s |
| NLinear | MLP y lineales | 0.882 | 45.44 | 1.829 | — | 32 s |
| LightGBM_m12 | Árboles | 0.884 | 43.98 | 1.705 | 1.055¹ | 32 s |
| TCN | Residuales y convolucionales | 0.884 | 45.79 | 1.640 | — | 146 s |
| DLinear_m3 | MLP y lineales | 0.888 | 45.25 | 1.910 | 0.986¹ | 28 s |
| MSTL | Baseline | 0.926 | 48.63 | 1.964 | 1.039 | (554 s de inferencia) |
| PatchTST_m3 | Atención | 1.006 | 47.72 | 1.849 | 1.082¹ | 80 s |
| SeasonalNaive24 | Baseline | 1.044 | 50.01 | 2.235 | 1.204 | — |
| DeepAR | Probabilísticos | 1.048 | 53.84 | 1.667 | — | 217 s |
| SeasonalNaive168 | Baseline | 1.074 | 56.63 | 1.623 | 1.080 | — |
| Naive | Baseline | 3.985 | 215.58 | 4.078 | 4.021 | — |

¹ El MASE con H = 168 de LightGBM, DLinear y PatchTST se midió con su configuración de referencia (30 meses), no con la elegida.

**Modelos probabilísticos:**

| Modelo | Pinball | Cobertura P10–P90 |
|---|---|---|
| ChronosBolt | 13.45 | 80.0% (nominal 80%) |
| DeepAR | 18.97 | 60.6% (intervalos demasiado angostos) |

**Otros resultados:**

| Resultado | Valor |
|---|---|
| Semillas (sección 8) | NHITS, el mejor modelo de DL: MASE **0.7786 ± 0.0038** con semillas 42, 7 y 2026 (0.7790, 0.7822, 0.7746) |
| Reentrenamiento mensual (sección 4) | NHITS: 0.7784 contra 0.7790 entrenado una vez. LightGBM: 0.8829 contra 0.8800. Los baselines locales (MSTL, seasonal naive) dan un pronóstico idéntico por construcción; el notebook lo comprueba con el seasonal naive. |
| Local contra global (5.3) | MLP_lr3e-4, el mejor propio, en 10 clientes (uno por decil de escala): local MASE 0.801 contra global 0.807, con 46 s para los 10 modelos locales. El global gana sobre todo en MT_130 (0.240 contra 0.651); el local, en los clientes grandes (MT_103, MT_204, MT_274). Tabla por cliente en el notebook. |
| Frontera de Pareto | Solo Chronos: no se entrena y tiene el menor MASE. |

### 3.2 Modelos de librería: `lab9/neural.py`

El notebook 2 escribe este módulo con `%%writefile`, igual que la Parte 1 con los suyos.

- **`NFModel(name, cls, L, H, exog, history_months, seed, **hiperparámetros)`** envuelve cualquier modelo de NeuralForecast con el contrato `fit/predict`.
  - En cada `fit` construye el modelo de nuevo, así que sirve para semillas y reentrenamiento.
  - Usa julio–septiembre como `val_size` para el early stopping.
  - Solo le entrega a `predict` las últimas L horas.
  - Si el modelo es probabilístico, devuelve P10, P50 y P90; el pronóstico puntual es la mediana.
- **`ChronosBolt`** usa `amazon/chronos-bolt-small` en bfloat16, con contexto de 2048 h y `predict_quantiles`.

**Tres problemas de NeuralForecast que el wrapper corrige.** Cualquiera que vuelva a usar NF en este panel debe tenerlos en cuenta:

1. **NF también normaliza las covariables por ventana.**
   - El seno/coseno del mes es constante dentro de una semana, así que su desviación sale ~1e-8. No es exactamente 0, que NF sí reemplaza por 1.
   - El día que cambia el mes, la covariable se divide entre ~1e-6. Eso daba pronósticos de ±50,000 kWh el 1 de octubre y el 1 de diciembre.
   - `scale_y_only` hace que NF normalice solo `y`. El calendario ya está en [-1, 1] y {0, 1}.
2. **El early stopping de NF no restaura los mejores pesos;** se queda con los del último paso. El callback `BestWeights` los restaura, como hace `TorchForecaster`.
3. **`step_size=24`.** Las ventanas de entrenamiento y validación terminan a las 23:00, igual que los orígenes del backtest. La validación baja de ~690k a ~29k ventanas.

**Configuración de cada modelo (L = 168, hasta 2000 pasos, validación cada 100, paciencia 5):**

| Modelo | Configuración |
|---|---|
| NHITS | Arquitectura por defecto, escalador `robust`, calendario como covariable futura |
| TCN | Kernel 3, dilataciones 1…64 (campo receptivo de 255 h), escalador `robust`, calendario |
| PatchTST | Parches de 24 h con paso 12, 3 capas, 16 cabezas, RevIN, lr 5e-4. No acepta covariables futuras. |
| TFT | Hidden 64, 4 cabezas, escalador `robust`, calendario en el decoder |
| DeepAR | LSTM 2×128 y t de Student. Ver la nota de abajo. |

**DeepAR.** Con los valores por defecto de NF dio MASE de validación 3.17; esa corrida queda registrada como `DeepAR_htrain1`. Se cambiaron tres cosas:

- `h_train=24`: con 1, solo aprende el paso siguiente y el error se dispara al realimentar 24 pasos.
- `valid_loss=MQLoss`: el MAE por defecto se mide sobre la media de las muestras, que con las colas de la t de Student es muy inestable.
- Lotes de 256 ventanas: por defecto valida con todas a la vez y no cabe en 8 GB.

### 3.3 Iteraciones de hiperparámetros (5.2, solo validación)

Están en `results/val_iters_parte2.csv`, con una fila por corrida (configuración, `spec`, parámetros, tiempos, memoria pico, `val_MAE`, `val_MASE`). Las curvas por epoch o por paso están en `results/runs.csv` + `results/history/`.

| Iteración | Corridas (MASE de validación) |
|---|---|
| L (Seq2Seq, 30 meses) | 48: 0.930 · **168: 0.864** · 336: 0.871 |
| Meses (Seq2Seq) | 3: 1.046 · 12: 0.894 · **30: 0.864** |
| Meses (LightGBM) | 3: 0.965 · **12: 0.907** · 30: 0.911 |
| Meses (DLinear) | **3: 0.873** · 12: 0.884 · 30: 0.883 |
| Meses (PatchTST) | **3: 1.000** · 12: 1.047 · 30: 1.055 |
| Normalización (MLP, Parte 1) | global: 3.228 · serie: 0.883 · **ventana: 0.877** |
| Otros (MLP; base 0.859) | hidden 1024: 0.870 · dropout 0.3: 0.896 · **lr 3e-4: 0.828** · batch 2048: 0.831 |
| `h_train` (DeepAR) | 1: 3.165 · **24: 1.109** |
| Referencias planas | SeasonalNaive24: 0.998 · ChronosBolt: 0.773 |

**Cómo se elige la configuración de cada modelo:**

- Se toma la de **menor MASE de validación**.
- Si no es la de referencia, se evalúa en prueba **una sola vez**: MLP_lr3e-4, LightGBM_m12, DLinear_m3 y PatchTST_m3.
- La tabla final conserva también las filas de referencia, con `elegida = False`.
- Ojo: DLinear_m3 y PatchTST_m3 ganaron en validación pero quedaron ligeramente peor en prueba que su versión de 30 meses.

### 3.4 Gráficas nuevas (`figures/`)

| Archivo | Contenido |
|---|---|
| `6_1_pareto_mase_tiempo.png` | Gráfica 1: MASE de prueba contra tiempo de entrenamiento (log), con la frontera de Pareto |
| `6_2_mae_por_paso_todos.png` | Gráfica 2: MAE por paso del horizonte, en 3 paneles (baselines y lineales, redes propias, librería) |
| `6_3_mase_vs_historia.png` | Gráfica 3: MASE de validación contra meses de historia, con el seasonal naive y Chronos como referencia |
| `5_2_curvas_perdida.png` | Curvas de pérdida y MASE por epoch: Seq2Seq L48/L168/L336/3 meses y LSTM_AR |
| `5_3_local_vs_global.png` | MASE local contra global por cliente |
| `8_navidad_modelos_parte2.png` | Consumo real contra pronosticado alrededor de Navidad |

### 3.5 Archivos de la Parte 2

| Archivo | Contenido |
|---|---|
| `preds/test_predictions.parquet` | **Entregable:** los 21 modelos de la tabla (`unique_id, ds, modelo, y_hat`) |
| `results/test_metrics.csv` | Tabla comparativa final (Parte 1 + Parte 2) con `elegida` y `MASE_H168` |
| `results/test_metrics_parte2.csv` | Métricas de prueba de los modelos nuevos |
| `results/test_metrics_H168.csv`, `test_refit_MS.csv`, `test_seeds.csv`, `local_vs_global.csv` | H = 168, reentrenamiento mensual, semillas y 5.3 |
| `results/test_by_horizon_parte2.csv`, `test_by_hour_parte2.csv`, `test_by_client_parte2.csv` | Desgloses por paso, hora y cliente de los modelos nuevos |
| `preds/<modelo>.parquet` | Pronósticos de cada corrida de prueba, incluidas las variantes `_H168`, `_refitMS` y `_seed*` |

### 3.6 Números de apoyo para la discusión (sección 8)

La última sección del notebook 2 reúne, para cada pregunta, la tabla que la responde:

- mejora contra SN24 y porcentaje de clientes en que gana;
- DLinear contra los Transformers con H = 24 y H = 168;
- MASE por paso y por familia;
- efecto de L, la normalización y la historia;
- Chronos contra los modelos entrenados y resultados probabilísticos;
- Navidad: MASE en la semana del 22 al 28 de diciembre y los días 24, 25 y 31, con y sin covariables;
- local contra global y semillas.

**Las respuestas escritas no están hechas.**

---

## 4. Lo que falta (entre los dos)

- **Sección 8 (discusión):** escribir las respuestas con base en las tablas de la última sección del notebook 2 (3.6 de este README).
- **Sección 3 (investigación):** `nn.LSTM`/`nn.GRU`, `nn.Conv1d` con dilatación y campo receptivo de la TCN, `nn.TransformerEncoderLayer` y codificación posicional, `cross_validation` de Nixtla (`n_windows`, `step_size`, `refit`), `predict_quantiles` de Chronos. Conceptos: local contra global, teacher forcing, RevIN, pinball loss, MASE contra WAPE.
- **PDF de máximo 5 páginas:** investigación, preparación y auditoría de fugas, tabla de iteraciones, estrategias multi-paso, tabla comparativa, resultados probabilísticos, gráficas, análisis y conclusiones. Al final va el **enlace al repositorio**.
- **Repositorio:** los notebooks comentados más `preds/test_predictions.parquet`. Los datos crudos de UCI no se suben (`data/raw/` está ignorado). `data/hourly.parquet` pesa 47 MB, por debajo del límite de 100 MB de GitHub.
  - Los `.parquet` de la Parte 2 suman ~550 MB. Ninguno pasa de 100 MB: el más grande es `ChronosBolt_H168.parquet`, con 92 MB, y el entregable pesa 61 MB.
  - Los auxiliares (`*_H168`, `*_seed*`, `*_refitMS`) se pueden dejar fuera del repo, porque sus métricas ya están en `results/`.

---

## Checklist

- [x] Instalar `neuralforecast` y `chronos-forecasting` y verificar que torch siga con CUDA
- [x] Wrapper de NeuralForecast + N-HiTS y TCN
- [x] PatchTST + TFT
- [x] DeepAR con P10/P50/P90 → pinball y cobertura
- [x] Chronos-Bolt zero-shot → pinball y cobertura
- [x] 5.2: iteraciones de L (48/168/336) y de meses de historia (3/12/30), más otros hiperparámetros, solo en validación
- [x] 5.2: gráfica de curvas de pérdida (≥ 3 iteraciones)
- [x] 5.3: 10 modelos locales contra el global
- [x] Reentrenamiento mensual del mejor DL y del mejor baseline
- [x] MASE con H = 168 donde aplique
- [x] Sección 6: tabla final ordenada por MASE + 3 gráficas
- [x] 3 semillas del mejor modelo → desviación estándar del MASE
- [x] `preds/test_predictions.parquet` con todos los modelos
- [ ] Sección 8: respuestas escritas (los números de apoyo están al final del notebook 2)
- [ ] Sección 3: investigación
- [ ] PDF de máximo 5 páginas con el enlace al repo
