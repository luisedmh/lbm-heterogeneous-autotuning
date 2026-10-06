#!/usr/bin/env python3
"""
analyze.py - resume los CSV de benchmark.py: MLUPS de la GPU y de la CPU por reparto.

Para cada precision y reparto (ny_gpu = filas que procesa la GPU) calcula la media
de las repeticiones de:
  - MLUPS_GPU : velocidad del lado GPU medida dentro del programa heterogeneo
  - MLUPS_CPU : velocidad del lado CPU medida dentro del programa heterogeneo
  - MLUPS_total, su variabilidad (CV %) y los tiempos por paso de cada lado.

Uso:
  python3 scripts/analyze.py results/benchmarks/XXXX.csv [mas.csv ...]
Escribe, junto al primer CSV, summary_<nombre>.csv (lo lee tambien el dashboard).
"""
import os
import sys
from pathlib import Path

# El analisis es minusculo: que numpy/BLAS no abra una piscina de hilos que compita con un benchmark
# que ocupe todos los nucleos.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse

import pandas as pd

NUM_COLS = ["NX", "NY", "ny_gpu", "ny_cpu", "steps", "warmup", "n_samples", "omp_threads", "mlups_total",
            "t_total_s", "t_step_ms_mean", "t_step_ms_std", "t_step_ms_median", "t_gpu_ms_mean",
            "t_gpu_ms_std", "t_gpu_ms_median", "t_cpu_ms_mean", "t_cpu_ms_std", "t_cpu_ms_median",
            "gpu_side_mlups", "cpu_side_mlups", "gpu_temp_start", "gpu_temp_max", "gpu_sm_clock_mean",
            "gpu_power_max", "cpu_mhz_start", "cpu_mhz_end", "order", "rep", "returncode"]


def load(paths):
    frames = []
    for p in paths:
        if Path(p).name.startswith(("summary_", "model_", "fit_")):
            continue            # ficheros derivados: un glob *.csv puede incluirlos
        df = pd.read_csv(p)
        if "returncode" not in df.columns or "ny_gpu" not in df.columns:
            continue            # no es un CSV del barrido de repartos
        df["source_file"] = Path(p).name
        frames.append(df)
    if not frames:
        sys.exit("No hay CSV del barrido de repartos entre los ficheros indicados.")
    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[(df["returncode"] == 0) & df["mlups_total"].notna()].copy()


def summarize(df):
    """Una fila por (precision, reparto) con las medias de las repeticiones."""
    agg = df.groupby(["precision", "ny_gpu"]).agg(
        n=("mlups_total", "size"),
        gpu_side_mlups=("gpu_side_mlups", "mean"), cpu_side_mlups=("cpu_side_mlups", "mean"),
        mlups_mean=("mlups_total", "mean"), mlups_std=("mlups_total", "std"),
        mlups_min=("mlups_total", "min"), mlups_max=("mlups_total", "max"),
        t_gpu_ms=("t_gpu_ms_mean", "mean"), t_cpu_ms=("t_cpu_ms_mean", "mean"),
        t_step_ms=("t_step_ms_mean", "mean"),
        gpu_temp_max=("gpu_temp_max", "mean"), gpu_clock=("gpu_sm_clock_mean", "mean"),
    ).reset_index()
    agg["mlups_cv_pct"] = 100.0 * agg["mlups_std"] / agg["mlups_mean"]
    agg["ny_cpu"] = df["NY"].iloc[0] - agg["ny_gpu"]
    return agg


def report(prec, d, summ):
    NX, NY = int(d["NX"].iloc[0]), int(d["NY"].iloc[0])
    print(f"\n{'=' * 78}\n PRECISION {prec}   ({len(d)} ejecuciones, {d['ny_gpu'].nunique()} repartos, {NX}x{NY})\n{'=' * 78}")
    cols = ["ny_gpu", "ny_cpu", "n", "gpu_side_mlups", "cpu_side_mlups", "mlups_mean", "mlups_cv_pct",
            "t_gpu_ms", "t_cpu_ms", "t_step_ms"]
    show = summ[summ["precision"] == prec][cols].rename(columns={
        "gpu_side_mlups": "MLUPS_GPU", "cpu_side_mlups": "MLUPS_CPU",
        "mlups_mean": "MLUPS_total", "mlups_cv_pct": "CV_%"})
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  MLUPS_GPU / MLUPS_CPU = velocidad de cada lado medida DENTRO del heterogeneo (celdas de ese lado /")
    print("  tiempo que ese lado tarda en un paso). Media de las repeticiones de cada reparto.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", type=Path)
    args = ap.parse_args()
    args.csv = [p for p in args.csv if not p.name.startswith(("summary_", "model_", "fit_"))]

    print(f"[analyze] leyendo {len(args.csv)} fichero(s) ...", flush=True)
    df = load(args.csv)
    if df.empty:
        sys.exit("No hay filas validas en los CSV (todas con returncode != 0 o sin BENCH_RESULT).")
    print(f"[analyze] {len(df)} ejecuciones validas", flush=True)
    summ = summarize(df)
    for prec in sorted(df["precision"].unique()):
        report(prec, df[df["precision"] == prec], summ)

    base = args.csv[0]
    stem = base.stem if len(args.csv) == 1 else base.stem + "_y_otros"
    summ_path = base.with_name(f"summary_{stem}.csv")
    summ.to_csv(summ_path, index=False)
    print(f"\n[ok] {summ_path}")


if __name__ == "__main__":
    main()
