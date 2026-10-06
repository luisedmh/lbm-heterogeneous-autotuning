# Medir los MLUPS reales de la GPU y de la CPU (heterogéneo estático)

## Qué se mide

La velocidad de **cada dispositivo por separado, pero dentro del programa heterogéneo**, con CPU y GPU
trabajando a la vez. Unidad: **MLUPS** (millones de actualizaciones de nodo por segundo).

- `src/hetero/lbm_heterogeneo_estatico_bench.cu`: heterogéneo con reparto fijo. La GPU calcula las primeras
  `--ny-gpu` filas y la CPU el resto. En cada paso mide por separado el lado GPU (`cudaEvent`: halo + kernel)
  y el lado CPU (reloj del host). Opciones: `--ny-gpu`, `--steps`, `--warmup`, `--obstacle`, `--dump`.
  El tamaño de malla (columnas × filas) se fija al compilar (`-DNX_SIZE`, `-DNY_SIZE`); lo hace `benchmark.py`.
- `scripts/benchmark.py`: repite los repartos N veces y guarda los datos (ver abajo).
- `scripts/bench_data.py`: lee y junta todos los CSV; lo usan `analyze.py` y el dashboard.
- `scripts/analyze.py`: resumen en terminal.
- `scripts/dashboard.py`: panel web con filtros.

## Variables de cada medida

| Variable | Valores | Cómo se elige |
|---|---|---|
| Precisión | FP64, FP32 | `--precision fp64 fp32` |
| Malla | columnas × filas, p. ej. `1000x1000` | `--grid 1000x1000 2000x1000` |
| Obstáculo | `cylinder` (defecto), `square`, `plate`, `none` | `--obstacle cylinder square` |
| Reparto | fracción de filas para la GPU (α) | `--alpha 0.6 0.66 0.7` (o `--ny-gpu` en filas absolutas) |
| Fecha | día en que se midió | automática |

Una sola orden puede cubrir varias combinaciones (precisión × malla × obstáculo); cada combinación genera su
propio CSV.

## Cómo se guardan los datos

```
results/benchmarks/
  2026-10-06/
    2026-10-06_102041_static_fp64_1000x1000_cylinder.csv     una fila por ejecución
    2026-10-06_102041_static_fp64_1000x1000_cylinder.json    ficha de la máquina y del comando
  2026-10-07/
    …
```

El nombre es `FECHA_HORA_etiqueta_precisión_columnasXfilas_obstáculo`. Cada fila del CSV lleva además `date`,
`timestamp`, `precision`, `NX`, `NY`, `obstacle`, `obstacle_cells`, `gpu_name`, `cpu_model` y `git_commit`, así que
el CSV se entiende solo y el dashboard puede juntar cualquier cantidad de ficheros. No hace falta guardar
resúmenes: se calculan al vuelo.

## Flujo

```bash
source .venv/bin/activate

# 1) validar (el cilindro por defecto debe seguir dando diferencia ~1e-14 con la referencia)
make -f Makefile.tfg build N=lbm_heterogeneo_estatico_bench.cu
(cd results/run && ../../bin/lbm_heterogeneo_estatico_bench --ny-gpu 823 --steps 3000 --dump \
   && python3 ../../tools/compara_resultados.py)

# 2) medir (en segundo plano por si se corta el SSH)
nohup python3 scripts/benchmark.py --precision fp64 > results/run/bench.log 2>&1 &
tail -f results/run/bench.log

# 3) ver
python3 scripts/analyze.py                       # terminal, todo lo guardado
streamlit run scripts/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
#   en el portátil:  ssh -N -L 8501:localhost:8501 luis@PC-Casa   ->  http://localhost:8501
```

## Para que la medida sea fiable

- Cierra todo lo demás que use GPU o CPU y no uses el PC durante la medida.
- Usa siempre ≥ 1000 pasos por ejecución.
- No mezcles en un mismo análisis medidas con distinto `steps`/`warmup` sin mirar esas columnas.
- Conserva el `.json` junto a cada CSV: contiene máquina, commit y comando exacto.
