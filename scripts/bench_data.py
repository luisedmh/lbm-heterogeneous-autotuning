"""
bench_data.py - lectura y normalizacion de TODOS los CSV de benchmark.py.

Lo usan analyze.py (terminal) y dashboard.py (web). Recorre results/benchmarks/
y sus subcarpetas (una por dia), junta todos los CSV en una unica tabla (una fila
por ejecucion) y anade las columnas derivadas que sirven para filtrar y agrupar:

    date      dia de la medida (AAAA-MM-DD)
    sweep_id  barrido al que pertenece (nombre del CSV sin extension)
    grid      tamano de malla "COLUMNASxFILAS", p. ej. 1000x1000
    obstacle  cylinder / square / plate / none
    alpha     fraccion de filas que procesa la GPU (ny_gpu / NY)
    config    "FP64 | 1000x1000 | cylinder"

Los CSV antiguos (anteriores a esta version) no tienen obstaculo ni fecha: se
consideran cilindro (era el unico caso) y la fecha se saca de su timestamp.
"""
import json
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO / "results" / "benchmarks"
DERIVED_PREFIXES = ("summary_", "model_", "fit_")

NUM_COLS = ["NX", "NY", "obstacle_cells", "ny_gpu", "ny_cpu", "steps", "warmup", "n_samples", "omp_threads",
            "mlups_total", "t_total_s", "t_step_ms_mean", "t_step_ms_std", "t_step_ms_median",
            "t_gpu_ms_mean", "t_gpu_ms_std", "t_gpu_ms_median", "t_cpu_ms_mean", "t_cpu_ms_std", "t_cpu_ms_median",
            "gpu_side_mlups", "cpu_side_mlups", "gpu_temp_start", "gpu_temp_end", "gpu_temp_max",
            "gpu_sm_clock_mean", "gpu_power_max", "cpu_mhz_start", "cpu_mhz_end", "order", "rep", "returncode"]


def find_csvs(root=DEFAULT_ROOT):
    """CSV de medidas (con columnas ny_gpu y returncode) bajo `root`, ficheros o carpetas."""
    root = Path(root)
    cands = [root] if root.is_file() else sorted(root.rglob("*.csv")) if root.exists() else []
    out = []
    for p in cands:
        if p.name.startswith(DERIVED_PREFIXES):
            continue
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception:
            continue
        if {"ny_gpu", "returncode"} <= cols:
            out.append(p)
    return out


def sidecar(csv_path):
    j = Path(csv_path).with_suffix(".json")
    try:
        return json.loads(j.read_text()) if j.exists() else {}
    except Exception:
        return {}


def load_all(paths=None, root=DEFAULT_ROOT):
    """Tabla unica con todas las ejecuciones validas. `paths`: lista de CSV/carpetas (defecto: `root`)."""
    files = []
    for p in (paths or [root]):
        files += find_csvs(p)
    files = sorted(set(files))
    frames = []
    for p in files:
        df = pd.read_csv(p)
        meta = sidecar(p)
        try:
            df["source_file"] = str(p.relative_to(REPO))
        except ValueError:
            df["source_file"] = str(p)
        for col, key, default in (("gpu_name", "gpu_name", "desconocida"), ("cpu_model", "cpu_model", "desconocida"),
                                  ("host", "hostname", "desconocido")):
            if col not in df.columns:
                df[col] = meta.get(key) or default
        if "sweep_id" not in df.columns:
            df["sweep_id"] = p.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["returncode"] == 0) & df["mlups_total"].notna()].copy()
    if df.empty:
        return df

    df["precision"] = df["precision"].astype(str).str.upper()
    # obstaculo: los CSV antiguos no lo tienen -> cilindro
    if "obstacle" not in df.columns:
        df["obstacle"] = "cylinder"
    df["obstacle"] = df["obstacle"].fillna("").astype(str).replace("", "cylinder")
    # fecha: columna 'date' si existe y, si no (o esta vacia), los 10 primeros caracteres del timestamp
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        d = d.fillna(ts.dt.normalize())
    else:
        d = ts.dt.normalize()
    df["date"] = d.dt.strftime("%Y-%m-%d")
    df["datetime"] = ts
    for c in ("gpu_name", "cpu_model", "host"):
        df[c] = df[c].fillna("").astype(str).replace("", "desconocida")
    df["NX"] = df["NX"].astype(int)
    df["NY"] = df["NY"].astype(int)
    df["ny_gpu"] = df["ny_gpu"].astype(int)
    df["grid"] = df["NX"].astype(str) + "x" + df["NY"].astype(str)
    df["alpha"] = (df["ny_gpu"] / df["NY"]).round(3)
    df["config"] = df["precision"] + " | " + df["grid"] + " | " + df["obstacle"]
    return df.sort_values(["datetime", "order"]).reset_index(drop=True)


def summarize(df, keys):
    """Una fila por combinacion de `keys` con medias de las repeticiones."""
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(list(keys), dropna=False).agg(
        n=("mlups_total", "size"),
        gpu_mlups=("gpu_side_mlups", "mean"), gpu_mlups_std=("gpu_side_mlups", "std"),
        cpu_mlups=("cpu_side_mlups", "mean"), cpu_mlups_std=("cpu_side_mlups", "std"),
        total_mlups=("mlups_total", "mean"), total_std=("mlups_total", "std"),
        total_min=("mlups_total", "min"), total_max=("mlups_total", "max"),
        t_gpu_ms=("t_gpu_ms_mean", "mean"), t_cpu_ms=("t_cpu_ms_mean", "mean"),
        t_step_ms=("t_step_ms_mean", "mean"),
        gpu_temp_max=("gpu_temp_max", "mean"),
    ).reset_index()
    agg["total_cv_pct"] = 100.0 * agg["total_std"] / agg["total_mlups"]
    return agg
