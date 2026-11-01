# Laboratorio 9: Series de tiempo y forecasting

Deep Learning y Sistemas Inteligentes (CC3092), Universidad del Valle de Guatemala.

**Autores:** Melisa Mendizabal (23778) y Renato Rojas (23813)

Este proyecto pronostica el consumo eléctrico de 315 clientes de Portugal para las siguientes 24 horas (pronóstico *day-ahead*). Comparamos baselines estadísticos, LightGBM, redes propias en PyTorch, modelos de NeuralForecast y el modelo fundacional Chronos-Bolt. Todos se evalúan con el mismo protocolo de backtesting.

## Resultados principales

Prueba de octubre a diciembre de 2014, con 92 orígenes diarios. La métrica principal es el MASE, donde un valor menor a 1 significa que el modelo le gana al seasonal naive de 24 horas.

| Modelo | Familia | MASE (H = 24) | MASE (H = 168) | MASE 22 al 28 dic | Entrenamiento |
|---|---|---|---|---|---|
| ChronosBolt (zero-shot) | Fundacional | **0.767** | **0.919** | 1.490 | No se entrena (31 s de inferencia) |
| NHITS | Residuales y convolucionales | 0.779 | 0.950 | **1.351** | 77 s |
| MLP (lr 3e-4) | MLP y lineales | 0.812 | 0.975 | 1.437 | 47 s |
| Seq2Seq | Encoder y decoder | 0.831 | 0.998 | 1.414 | 1,223 s |
| LSTM autorregresiva | Recurrentes | 0.832 |  | 1.394 | 1,292 s |
| TFT | Atención | 0.833 |  | 1.434 | 287 s |
| LightGBM (12 meses) | Árboles | 0.884 | 1.055 | 1.705 | 32 s |
| DLinear (3 meses) | MLP y lineales | 0.888 | 0.986 | 1.910 | 28 s |
| MSTL | Baseline | 0.926 | 1.039 | 1.964 | 554 s de inferencia |
| PatchTST (3 meses) | Atención | 1.006 | 1.082 | 1.849 | 80 s |
| Seasonal naive 24 | Baseline | 1.044 | 1.204 | 2.235 |  |
| DeepAR | Probabilísticos | 1.048 |  | 1.667 | 217 s |

La tabla completa, con 21 modelos y todas las métricas (MAE, RMSE, WAPE, MAPE, pinball y cobertura), está en [results/test_metrics.csv](results/test_metrics.csv).

**Lo más importante:**

- **NHITS es el mejor modelo entrenado.** Mejora el MASE del seasonal naive en 25%. Con 3 semillas distintas da 0.779 ± 0.004, así que el resultado es estable.
- **Chronos-Bolt queda primero sin entrenar nada**, y además da los mejores intervalos (cobertura P10 a P90 de 80.0%, igual a la nominal). Hay que leerlo con cuidado: el dataset de electricidad de UCI aparece en los datos de preentrenamiento de Chronos (Ansari et al., 2024), así que es posible que ya haya visto estas series.
- **Navidad es el periodo más difícil.** En la semana del 22 al 28 de diciembre el MASE de los modelos sube entre 51% y 115% (sin contar Naive). Los modelos con covariables de calendario empeoran menos (77%) que los que no las usan (97%).
- **Un modelo global le gana a 10 modelos locales** solo en los clientes medianos y grandes. En los clientes pequeños el modelo local es mejor.

## Datos y protocolo

- **Datos:** [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) de UCI. Son 370 clientes medidos cada 15 minutos, que agregamos a consumo por hora (kWh).
- **Panel:** nos quedamos con 315 clientes que tienen datos desde enero de 2012 y no tienen más de 7 días completos en cero.
- **Particiones:**
  - Entrenamiento: enero de 2012 a junio de 2014.
  - Validación: julio a septiembre de 2014. Solo se usa para early stopping y para elegir hiperparámetros.
  - Prueba: octubre a diciembre de 2014.
- **Backtesting:** una sola función `backtest()` para todos los modelos. En cada origen (a las 00:00) el modelo solo ve datos hasta las 23:00 del día anterior y pronostica las 24 horas siguientes.
- **Preparación:**
  - El cambio de horario se corrige: en marzo se interpola la hora que falta y en octubre la hora repetida se divide entre 2.
  - Cada ventana se normaliza con su propia media y desviación (tipo RevIN).
  - El calendario se codifica con seno y coseno (hora, día de la semana, mes) más una variable de feriado de Portugal.
- **Métricas:** MASE (con la escala del seasonal naive de 24 horas, calculada solo con entrenamiento), MAE, RMSE, WAPE y MAPE. Para los modelos probabilísticos también pinball loss y cobertura del intervalo P10 a P90.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| [parte1_datos_y_modelos.ipynb](parte1_datos_y_modelos.ipynb) | Exploración y preparación de la serie, investigación (sección 3), protocolo, métricas, baselines, LightGBM y redes propias (MLP, DLinear, NLinear, LSTM, Seq2Seq) |
| [parte2_modelos_y_analisis.ipynb](parte2_modelos_y_analisis.ipynb) | Modelos de NeuralForecast (NHITS, TCN, PatchTST, TFT, DeepAR), Chronos-Bolt, iteraciones de hiperparámetros, local contra global, comparación final y discusión (sección 8) |
| `informe_lab9.docx` | Informe escrito del laboratorio |
| [lab9/](lab9/) | Código compartido: carga de datos, `Dataset` de PyTorch, backtest, métricas, baselines, modelos y entrenamiento |
| [data/](data/) | Panel limpio por hora (`hourly.parquet`), clientes elegidos, feriados y estadísticas por cliente |
| [preds/](preds/) | Pronósticos de prueba. El entregable es `test_predictions.parquet`, con columnas `unique_id, ds, modelo, y_hat` |
| [results/](results/) | Métricas de prueba, iteraciones de validación, registro de cada entrenamiento y curvas de pérdida por epoch |
| [figures/](figures/) | Todas las gráficas de los notebooks |
| [checkpoints/](checkpoints/) | Pesos de las redes propias |

**Sobre `lab9/`:** los notebooks escriben estos archivos con `%%writefile`. El notebook 1 genera todos menos `neural.py`, que lo genera el notebook 2. Si cambias un archivo a mano, cámbialo también en su celda, porque se sobrescribe al volver a correr el notebook.

## Cómo correrlo

Se necesita Python 3.11 y una GPU con CUDA. Nosotros usamos una RTX 4050 para la Parte 1 y una RTX 4060 Laptop para la Parte 2.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name lab9 --display-name "Python (lab9)"
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Si el último comando imprime `False`, pip instaló la versión de torch sin CUDA. En ese caso hay que reinstalar torch con la versión de CUDA de tu GPU desde [pytorch.org](https://pytorch.org/get-started/locally/).

Después se corren los notebooks en orden:

1. **`parte1_datos_y_modelos.ipynb`** descarga los datos de UCI la primera vez (en `data/raw/`, que no se sube al repo), los prepara y entrena los modelos de la Parte 1. Tarda aproximadamente 1.5 horas.
2. **`parte2_modelos_y_analisis.ipynb`** usa los archivos que deja la Parte 1. Si ya existen los resultados en `results/` y `preds/`, los carga en lugar de volver a entrenar, y termina en unos 2 minutos. Con `RECOMPUTE = True` en la celda de configuración entrena todo de nuevo, lo cual toma de 3 a 4 horas.

Para revisar que todo corre sin esperar horas, se puede definir la variable de entorno `LAB9_QUICK=1` antes de abrir Jupyter. Así se usan pocos clientes y pocas epochs. Los números que salen de esa corrida no son válidos.

**Versiones.** `requirements.txt` tiene las versiones de la Parte 1. La Parte 2 corrió con `torch 2.13.0+cu126`, `neuralforecast 3.3.0`, `chronos-forecasting 2.3.2`, `pandas 2.2.2` y `numpy 1.26.4`. La semilla de todos los experimentos es 42.

## Referencias

- Trindade, A. (2015). *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository.
- Ansari, A. F. et al. (2024). *Chronos: Learning the Language of Time Series*.
- Challu, C. et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*.
- Zeng, A. et al. (2023). *Are Transformers Effective for Time Series Forecasting?*
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*.
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Salinas, D. et al. (2020). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*.
- Kim, T. et al. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*.
- Hyndman, R. y Koehler, A. (2006). *Another look at measures of forecast accuracy*.
