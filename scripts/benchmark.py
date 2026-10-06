#!/usr/bin/env python3
"""
benchmark.py - mide la velocidad real de CPU y GPU dentro del programa
heterogeneo ESTATICO (reparto fijo), con ambos dispositivos trabajando a la vez.

Cada invocacion puede cubrir varias configuraciones (producto de --precision x
--grid x --obstacle). Para CADA configuracion hace un barrido de repartos
(filas para la GPU) con repeticiones y guarda su propio CSV:

    results/benchmarks/<AAAA-MM-DD>/<AAAA-MM-DD>_<HHMMSS>_<etiqueta>_<prec>_<NXxNY>_<obstaculo>.csv
    results/benchmarks/<AAAA-MM-DD>/<mismo nombre>.json      (ficha de la maquina y del comando)

Asi cada medida queda en la carpeta del dia en que se hizo y el nombre dice
que es. Cada fila del CSV es UNA ejecucion y lleva su fecha y hora, la
precision, el tamano de malla, el obstaculo, la GPU/CPU usadas, etc., de modo
que el dashboard puede juntar todos los CSV y filtrar por cualquiera de ellos.

Detalles de metodologia:
  - El ORDEN de las ejecuciones se baraja (semilla fija, --seed) dentro de cada
    configuracion: la deriva termica no se confunde con el efecto del reparto.
  - Antes de medir cada configuracion se hace una ejecucion de calentamiento
    completa que se descarta (--no-warmup-run para saltarla).
  - Entre ejecuciones se espera --cooldown segundos.
  - El ejecutable se compila una vez por (precision, tamano) y se reutiliza.

Ejemplos:
  python3 scripts/benchmark.py                                   # FP64, 1000x1000, cilindro, 12 repartos x 10 reps
  python3 scripts/benchmark.py --precision fp64 fp32             # las dos precisiones
  python3 scripts/benchmark.py --grid 1000x1000 2000x1000        # varios tamanos (columnasxfilas)
  python3 scripts/benchmark.py --obstacle cylinder square plate  # varios obstaculos
  python3 scripts/benchmark.py --alpha 0.62 0.64 0.66 0.68 0.70  # barrido fino (fraccion de filas para la GPU)
  python3 scripts/benchmark.py --reps 2 --ny-gpu 700 --steps 300 --cooldown 1   # prueba rapida
"""
import argparse
import csv
import datetime as dt
import itertools
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "hetero" / "lbm_heterogeneo_estatico_bench.cu"
OBSTACLES = ["cylinder", "square", "plate", "none"]
# Repartos por defecto, como FRACCION de filas para la GPU (para 1000 filas: 300 400 ... 823 ... 950)
DEFAULT_ALPHAS = [0.30, 0.40, 0.50, 0.60, 0.66, 0.70, 0.75, 0.80, 0.823, 0.85, 0.90, 0.95]

CSV_FIELDS = [
    # identificacion
    "run_id", "sweep_id", "order", "rep", "date", "timestamp", "label",
    # configuracion
    "precision", "NX", "NY", "obstacle", "obstacle_cells", "ny_gpu", "ny_cpu", "steps", "warmup", "n_samples",
    "omp_threads",
    # resultados
    "mlups_total", "t_total_s",
    "t_step_ms_mean", "t_step_ms_std", "t_step_ms_median",
    "t_gpu_ms_mean", "t_gpu_ms_std", "t_gpu_ms_median",
    "t_cpu_ms_mean", "t_cpu_ms_std", "t_cpu_ms_median",
    "gpu_side_mlups", "cpu_side_mlups",
    # contexto de la medida
    "sample_ms", "gpu_temp_start", "gpu_temp_end", "gpu_temp_max", "gpu_sm_clock_mean", "gpu_power_max",
    "cpu_mhz_start", "cpu_mhz_end",
    "gpu_name", "cpu_model", "host",
    # trazabilidad
    "git_commit", "git_dirty", "binary", "returncode",
]


# ----------------------------------------------------------------------------
# Utilidades de sistema
# ----------------------------------------------------------------------------
def run_quiet(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def nvidia_query(fields):
    """Devuelve una lista de floats/strings con los campos pedidos, o None."""
    if not shutil.which("nvidia-smi"):
        return None
    out = run_quiet(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    return [x.strip() for x in out.splitlines()[0].split(",")]


def tofloat(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cpu_mhz_mean():
    """Frecuencia media de los nucleos (MHz), leida de /proc/cpuinfo."""
    try:
        vals = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.lower().startswith("cpu mhz")]
        return round(sum(vals) / len(vals), 1) if vals else None
    except Exception:
        return None


def git_info():
    commit = run_quiet(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]) or "nogit"
    dirty = bool(run_quiet(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    return commit, dirty


def machine_info():
    gpu = nvidia_query("name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit") or []
    cpu_model = ""
    try:
        for l in open("/proc/cpuinfo"):
            if l.lower().startswith("model name"):
                cpu_model = l.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    governor = ""
    try:
        governor = open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
    except Exception:
        pass
    nvcc = run_quiet(["nvcc", "--version"]).splitlines()
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_logical_cores": os.cpu_count(),
        "cpu_governor": governor,
        "gpu_name": gpu[0] if len(gpu) > 0 else None,
        "gpu_driver": gpu[1] if len(gpu) > 1 else None,
        "gpu_memory_total_mib": gpu[2] if len(gpu) > 2 else None,
        "gpu_max_sm_clock_mhz": gpu[3] if len(gpu) > 3 else None,
        "gpu_max_mem_clock_mhz": gpu[4] if len(gpu) > 4 else None,
        "gpu_power_limit_w": gpu[5] if len(gpu) > 5 else None,
        "nvcc": nvcc[-1] if nvcc else None,
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "python": sys.version.split()[0],
    }


class GpuSampler(threading.Thread):
    """Muestrea temperatura / reloj SM / potencia de la GPU mientras corre un programa.

    Usa UN solo proceso `nvidia-smi -lms N` que va escribiendo lineas (en vez de
    lanzar un proceso nuevo en cada muestra), para perturbar lo minimo posible
    la medida: los 12 hilos de OpenMP usan todos los nucleos y cualquier proceso
    extra compite con ellos. Con period_ms = 0 no muestrea nada.
    """

    def __init__(self, period_ms=500):
        super().__init__(daemon=True)
        self.period_ms = period_ms
        self.proc = None
        self.temps, self.clocks, self.powers = [], [], []

    def run(self):
        if self.period_ms <= 0 or not shutil.which("nvidia-smi"):
            return
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                 "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in self.proc.stdout:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 3:
                    continue
                t, c, p = (tofloat(v) for v in parts[:3])
                if t is not None: self.temps.append(t)
                if c is not None: self.clocks.append(c)
                if p is not None: self.powers.append(p)
        except Exception:
            pass

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.join(timeout=3)
        mean = lambda v: round(sum(v) / len(v), 1) if v else None
        return {
            "gpu_temp_max": max(self.temps) if self.temps else None,
            "gpu_sm_clock_mean": mean(self.clocks),
            "gpu_power_max": max(self.powers) if self.powers else None,
        }


# ----------------------------------------------------------------------------
# Compilacion y ejecucion
# ----------------------------------------------------------------------------
def binary_path(precision, nx, ny):
    return REPO / "bin" / f"lbm_het_bench_{precision}_{nx}x{ny}"


def build(precision, nx, ny):
    """Compila (si hace falta) el ejecutable para esta precision y tamano de malla."""
    out = binary_path(precision, nx, ny)
    if out.exists() and out.stat().st_mtime >= SRC.stat().st_mtime:
        print(f"[build] {out.name}: ya compilado, se reutiliza", flush=True)
        return out
    out.parent.mkdir(exist_ok=True)
    cmd = ["nvcc", "-O3", "-arch=native", f"-DNX_SIZE={nx}", f"-DNY_SIZE={ny}"]
    if precision == "fp32":
        cmd.append("-DUSE_SINGLE_PRECISION")
    cmd += ["-Xcompiler", "-fopenmp", "-lgomp", str(SRC), "-o", str(out)]
    print("[build]", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=REPO).returncode != 0:
        sys.exit("ERROR: fallo la compilacion (revisa que nvcc esta en el PATH).")
    return out


BENCH_RE = re.compile(r"^BENCH_RESULT\s+(.*)$", re.M)


def parse_bench(stdout):
    m = BENCH_RE.search(stdout)
    if not m:
        return None
    d = {}
    for kv in m.group(1).split():
        k, _, v = kv.partition("=")
        d[k] = v
    return d


def run_once(binary, ny_gpu, obstacle, steps, warmup, threads, workdir, sample_ms=500):
    env = os.environ.copy()
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    cmd = [str(binary), "--ny-gpu", str(ny_gpu), "--obstacle", obstacle,
           "--steps", str(steps), "--warmup", str(warmup)]
    sampler = GpuSampler(sample_ms)
    pre = nvidia_query("temperature.gpu")
    temp_start = tofloat(pre[0]) if pre else None
    mhz_start = cpu_mhz_mean()
    sampler.start()
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    gpu_stats = sampler.stop()
    mhz_end = cpu_mhz_mean()
    post = nvidia_query("temperature.gpu")
    temp_end = tofloat(post[0]) if post else None
    if gpu_stats["gpu_temp_max"] is None:   # sin muestreo: al menos inicio/fin
        cands = [t for t in (temp_start, temp_end) if t is not None]
        gpu_stats["gpu_temp_max"] = max(cands) if cands else None
    res = parse_bench(proc.stdout)
    extra = {"gpu_temp_start": temp_start, "gpu_temp_end": temp_end, "sample_ms": sample_ms,
             "cpu_mhz_start": mhz_start, "cpu_mhz_end": mhz_end, **gpu_stats}
    return proc, res, extra


def parse_grid(text):
    m = re.fullmatch(r"(\d+)[xX](\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"tamano de malla invalido '{text}': usa COLUMNASxFILAS, p. ej. 1000x1000")
    return int(m.group(1)), int(m.group(2))


def splits_for(ny_total, ny_gpu_list, alphas):
    """Lista ordenada y sin repetidos de filas para la GPU, validas para este NY."""
    if ny_gpu_list:
        rows = list(ny_gpu_list)
    else:
        rows = [round(a * ny_total) for a in (alphas or DEFAULT_ALPHAS)]
    rows = sorted({r for r in rows if 2 <= r <= ny_total - 2})
    if not rows:
        sys.exit(f"ERROR: ningun reparto valido para NY={ny_total} (debe estar entre 2 y {ny_total - 2}).")
    return rows


def sweep_one(args, precision, grid, obstacle, binary, machine, commit, dirty):
    """Un barrido de repartos para UNA configuracion; escribe su CSV y su JSON."""
    nx, ny_total = grid
    rows = splits_for(ny_total, args.ny_gpu, args.alpha)
    now = dt.datetime.now()
    date = now.strftime("%Y-%m-%d")
    sweep_id = f"{date}_{now.strftime('%H%M%S')}_{args.label}_{precision}_{nx}x{ny_total}_{obstacle}"
    outdir = args.outdir / date
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{sweep_id}.csv"
    meta = dict(machine)
    meta.update({"sweep_id": sweep_id, "command": " ".join(sys.argv),
                 "config": {"precision": precision.upper(), "NX": nx, "NY": ny_total, "obstacle": obstacle,
                            "splits_ny_gpu": rows, "reps": args.reps, "steps": args.steps, "warmup": args.warmup},
                 "args": {k: str(v) for k, v in vars(args).items()},
                 "git_commit": commit, "git_dirty": dirty, "started": now.isoformat(timespec="seconds")})
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    workdir = REPO / "results" / "run"
    workdir.mkdir(parents=True, exist_ok=True)
    plan = [(ny, rep) for ny in rows for rep in range(1, args.reps + 1)]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(plan)
    total = len(plan)
    print(f"\n[plan] {sweep_id}\n[plan] {total} ejecuciones ({len(rows)} repartos x {args.reps} repeticiones), "
          f"{args.steps} pasos cada una", flush=True)
    print(f"[plan] CSV: {csv_path}", flush=True)

    if not args.no_warmup_run:
        ny_w = rows[len(rows) // 2]
        print(f"[warmup] ejecucion de calentamiento (ny_gpu={ny_w}), se descarta ...", flush=True)
        proc, res, _ = run_once(binary, ny_w, obstacle, args.steps, args.warmup, args.threads, workdir, args.sample_ms)
        if res is None:
            sys.exit("ERROR: el programa no produjo BENCH_RESULT.\n--- stdout ---\n"
                     f"{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        time.sleep(args.cooldown)

    t0 = time.time()
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, (ny, rep) in enumerate(plan, start=1):
            proc, res, extra = run_once(binary, ny, obstacle, args.steps, args.warmup, args.threads, workdir, args.sample_ms)
            stamp = dt.datetime.now()
            row = {k: "" for k in CSV_FIELDS}
            row.update({"run_id": f"{sweep_id}-{i:04d}", "sweep_id": sweep_id, "order": i, "rep": rep,
                        "date": stamp.strftime("%Y-%m-%d"), "timestamp": stamp.isoformat(timespec="seconds"),
                        "label": args.label, "obstacle": obstacle,
                        "gpu_name": machine.get("gpu_name") or "", "cpu_model": machine.get("cpu_model") or "",
                        "host": machine.get("hostname") or "",
                        "git_commit": commit, "git_dirty": int(dirty), "binary": binary.name,
                        "returncode": proc.returncode})
            row.update(extra)
            if res:
                for k in CSV_FIELDS:
                    if k in res:
                        row[k] = res[k]
            else:
                print(f"  ! ejecucion {i} sin BENCH_RESULT (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
            w.writerow(row)
            fh.flush()
            eta = (time.time() - t0) / i * (total - i)
            print(f"[{i:>3}/{total}] ny_gpu={ny:<5} rep={rep:<2} "
                  f"MLUPS GPU={row['gpu_side_mlups']:<9} CPU={row['cpu_side_mlups']:<9} total={row['mlups_total']:<9} "
                  f"T_gpu={row['gpu_temp_max'] or '?'}C  (quedan ~{eta / 60:.1f} min)", flush=True)
            if i < total:
                time.sleep(args.cooldown)
    print(f"[ok] guardado: {csv_path.relative_to(REPO) if csv_path.is_relative_to(REPO) else csv_path}", flush=True)
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", nargs="+", choices=["fp64", "fp32"], default=["fp64"],
                    help="precision(es) a medir (defecto fp64)")
    ap.add_argument("--grid", nargs="+", type=parse_grid, default=[(1000, 1000)], metavar="COLxFIL",
                    help="tamano(s) de malla COLUMNASxFILAS (defecto 1000x1000)")
    ap.add_argument("--obstacle", nargs="+", choices=OBSTACLES, default=["cylinder"],
                    help="tipo(s) de obstaculo (defecto cylinder)")
    ap.add_argument("--ny-gpu", type=int, nargs="+", default=None,
                    help="repartos como numero ABSOLUTO de filas para la GPU (solo tiene sentido con un unico tamano)")
    ap.add_argument("--alpha", type=float, nargs="+", default=None,
                    help="repartos como FRACCION de filas para la GPU (0-1); valen para cualquier tamano. "
                         "Defecto: " + " ".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--reps", type=int, default=10, help="repeticiones por reparto (defecto 10)")
    ap.add_argument("--steps", type=int, default=1000, help="pasos LBM por ejecucion (defecto 1000)")
    ap.add_argument("--warmup", type=int, default=100, help="pasos iniciales descartados de las estadisticas")
    ap.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS (defecto: todos los hilos)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="segundos de pausa entre ejecuciones")
    ap.add_argument("--sample-ms", type=int, default=500,
                    help="cada cuantos ms se muestrea temperatura/reloj de la GPU (0 = no muestrear)")
    ap.add_argument("--label", default="static", help="etiqueta para el nombre del CSV")
    ap.add_argument("--seed", type=int, default=12345, help="semilla para barajar el orden")
    ap.add_argument("--no-shuffle", action="store_true", help="no barajar el orden (no recomendado)")
    ap.add_argument("--no-warmup-run", action="store_true", help="saltar la ejecucion de calentamiento previa")
    ap.add_argument("--no-build", action="store_true", help="no compilar: usa los ejecutables que ya existan en bin/")
    ap.add_argument("--outdir", type=Path, default=REPO / "results" / "benchmarks")
    args = ap.parse_args()

    if args.ny_gpu and len(args.grid) > 1:
        sys.exit("ERROR: --ny-gpu (filas absolutas) no vale con varios tamanos; usa --alpha (fracciones).")

    configs = list(itertools.product(args.precision, args.grid, args.obstacle))
    print(f"[plan] {len(configs)} configuracion(es): "
          + ", ".join(f"{p.upper()} {g[0]}x{g[1]} {o}" for p, g, o in configs), flush=True)

    # compilar (o localizar) un ejecutable por (precision, tamano)
    binaries = {}
    for p, g, _ in configs:
        if (p, g) in binaries:
            continue
        b = binary_path(p, *g) if args.no_build else build(p, *g)
        if not b.exists():
            sys.exit(f"ERROR: no existe el ejecutable {b} (quita --no-build para compilarlo)")
        binaries[(p, g)] = b

    commit, dirty = git_info()
    machine = machine_info()
    written = []
    for n, (p, g, o) in enumerate(configs, start=1):
        print(f"\n########## configuracion {n}/{len(configs)}: {p.upper()}  {g[0]}x{g[1]}  obstaculo={o} ##########", flush=True)
        written.append(sweep_one(args, p, g, o, binaries[(p, g)], machine, commit, dirty))
        if n < len(configs):
            time.sleep(args.cooldown)

    print("\n[ok] terminado. Ficheros creados:")
    for w in written:
        print("   ", w)
    print("[ok] para verlos:  streamlit run scripts/dashboard.py   (o  python3 scripts/analyze.py)")


if __name__ == "__main__":
    main()
