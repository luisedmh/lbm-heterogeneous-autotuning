# Medir los MLUPS reales de la GPU y de la CPU (heterogeneo estatico)

## Que se mide

La velocidad de **cada dispositivo por separado, pero dentro del programa heterogeneo**, con CPU y GPU
trabajando a la vez (compitiendo por memoria y sistema). Unidad: **MLUPS** (millones de actualizaciones de
nodo por segundo).

- `src/hetero/lbm_heterogeneo_estatico_bench.cu`: heterogeneo con **reparto fijo** elegido por linea de comandos
  (`--ny-gpu`: filas que procesa la GPU; la CPU hace el resto). En cada paso mide por separado
  - el lado **GPU**: `cudaEvent` desde antes de subir el halo hasta despues de bajarlo (incluye el kernel);
  - el lado **CPU**: reloj del host alrededor de `cpu_fused_step`.

  MLUPS de un lado = celdas de ese lado / tiempo medio de ese lado por paso (se descartan los `--warmup`
  primeros pasos). Salen en la linea `BENCH_RESULT` como `gpu_side_mlups` y `cpu_side_mlups`.
- `scripts/benchmark.py` repite cada reparto N veces (orden barajado) y guarda un CSV con temperatura y reloj de la GPU.
- `scripts/analyze.py` resume en terminal: MLUPS de GPU y de CPU por reparto (media de las repeticiones).
- `scripts/dashboard.py` muestra los mismos resultados en una pagina web (Streamlit).

## Flujo

```bash
source .venv/bin/activate

# 1) validar el programa instrumentado (FP64, 3000 pasos, contra la referencia)
make -f Makefile.tfg run N=lbm_gpu_original_con_dump.cu          # si no existe results/run/f_final_original.bin
make -f Makefile.tfg build N=lbm_heterogeneo_estatico_bench.cu
(cd results/run && ../../bin/lbm_heterogeneo_estatico_bench --ny-gpu 823 --steps 3000 --dump \
   && python3 ../../tools/compara_resultados.py)

# 2) barrido de repartos (12 repartos x 10 repeticiones, ~12-15 min); en segundo plano por si se corta el SSH
nohup python3 scripts/benchmark.py --no-build --precision fp64 --reps 10 --label static \
  > results/run/bench_static.log 2>&1 &
tail -f results/run/bench_static.log

# 3) resultados en terminal (sustituye por el nombre real del CSV)
python3 scripts/analyze.py results/benchmarks/<fichero_static>.csv

# 4) dashboard
streamlit run scripts/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
#   en el portatil:  ssh -N -L 8501:localhost:8501 luis@PC-Casa   ->  http://localhost:8501
```

## Para que la medida sea fiable

- Cierra todo lo demas que use GPU o CPU y no uses el PC durante la medida.
- Usa siempre >= 1000 pasos por ejecucion (300 son pocos).
- El muestreo de la GPU (`--sample-ms`, 500 por defecto) no perturba la medida (comprobado con A/B); `0` lo desactiva.
- Conserva el `.json` que se guarda junto a cada CSV: contiene maquina, commit y comando exacto.
