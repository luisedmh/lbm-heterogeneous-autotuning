#!/usr/bin/env python3
"""
analyze.py - resume en terminal los MLUPS de GPU y de CPU de todas las medidas guardadas.

Sin argumentos lee TODO results/benchmarks/ (todas las carpetas de dias). Para cada
configuracion (precision | tamano de malla | obstaculo) imprime la media por reparto.

Uso:
  python3 scripts/analyze.py                                   # todo lo guardado
  python3 scripts/analyze.py --precision FP64 --grid 1000x1000 # solo una parte
  python3 scripts/analyze.py --obstacle square --desde 2026-10-07
  python3 scripts/analyze.py results/benchmarks/2026-10-06     # solo una carpeta (o un CSV)
  python3 scripts/analyze.py --out results/resumen.csv         # ademas guarda la tabla resumen
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="CSV o carpetas (defecto: results/benchmarks)")
    ap.add_argument("--precision", nargs="+", help="p. ej. FP64 FP32")
    ap.add_argument("--grid", nargs="+", help="p. ej. 1000x1000")
    ap.add_argument("--obstacle", nargs="+", help="p. ej. cylinder square")
    ap.add_argument("--desde", help="fecha minima AAAA-MM-DD")
    ap.add_argument("--hasta", help="fecha maxima AAAA-MM-DD")
    ap.add_argument("--out", type=Path, help="guardar aqui la tabla resumen (CSV)")
    args = ap.parse_args()

    df = bd.load_all(args.paths or None)
    if df.empty:
        sys.exit("No hay medidas validas. Genera datos con:  python3 scripts/benchmark.py")
    if args.precision:
        df = df[df["precision"].isin([p.upper() for p in args.precision])]
    if args.grid:
        df = df[df["grid"].isin(args.grid)]
    if args.obstacle:
        df = df[df["obstacle"].isin(args.obstacle)]
    if args.desde:
        df = df[df["date"] >= args.desde]
    if args.hasta:
        df = df[df["date"] <= args.hasta]
    if df.empty:
        sys.exit("Ninguna medida cumple esos filtros.")

    print(f"[analyze] {len(df)} ejecuciones, {df['sweep_id'].nunique()} barridos, "
          f"{df['config'].nunique()} configuraciones, dias {df['date'].min()} .. {df['date'].max()}")
    summ = bd.summarize(df, ["precision", "grid", "obstacle", "NX", "NY", "ny_gpu"])
    cols = ["ny_gpu", "n", "gpu_mlups", "cpu_mlups", "total_mlups", "total_cv_pct", "t_gpu_ms", "t_cpu_ms", "t_step_ms"]
    names = {"gpu_mlups": "MLUPS_GPU", "cpu_mlups": "MLUPS_CPU", "total_mlups": "MLUPS_total", "total_cv_pct": "CV_%"}
    for (prec, grid, obs), g in summ.groupby(["precision", "grid", "obstacle"]):
        dd = df[(df["precision"] == prec) & (df["grid"] == grid) & (df["obstacle"] == obs)]
        print(f"\n{'=' * 78}\n {prec} | malla {grid} (columnas x filas) | obstaculo {obs}   "
              f"[{len(dd)} ejecuciones, dias: {', '.join(sorted(dd['date'].unique()))}]\n{'=' * 78}")
        print(g.sort_values("ny_gpu")[cols].rename(columns=names).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nMLUPS_GPU / MLUPS_CPU = velocidad de cada lado medida DENTRO del heterogeneo; media de las repeticiones.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summ.to_csv(args.out, index=False)
        print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
