#!/usr/bin/env python3
"""
benchmark.py - mide la velocidad real de CPU y GPU dentro del programa
heterogeneo ESTATICO (reparto fijo), con ambos dispositivos trabajando a la vez.

Que hace:
  1. Compila src/hetero/lbm_heterogeneo_estatico_bench.cu (via Makefile.tfg).
  2. Para cada reparto (--ny-gpu) y cada repeticion (--reps) ejecuta el
     programa y lee su linea BENCH_RESULT: tiempo del lado GPU, del lado CPU
     y del paso completo, en cada paso, con media/desviacion/mediana.
  3. Registra, ademas, en cada ejecucion: temperatura, reloj y potencia de la
     GPU (muestreados mientras corre), frecuencia de CPU, commit de git, etc.
  4. Guarda UNA fila por ejecucion en results/benchmarks/<fecha>_<etiqueta>.csv
     (se escribe fila a fila: si interrumpes, no pierdes lo ya medido) y los
     metadatos de la maquina en un .json al lado.

Detalles de metodologia:
  - El ORDEN de las ejecuciones se baraja (semilla fija, --seed): asi la
    deriva termica no se confunde con el efecto del reparto.
  - Antes de medir se hace una ejecucion de calentamiento completa que se
    descarta (--no-warmup-run para saltarla).
  - Entre ejecuciones se espera --cooldown segundos.

Ejemplo (FP64, el caso por defecto):
  python3 scripts/benchmark.py --precision fp64 --reps 10
Prueba rapida:
  python3 scripts/benchmark.py --reps 2 --ny-gpu 700 823 --steps 300 --cooldown 1
"""
import argparse
import csv
import datetime as dt
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
SRC_NAME = "lbm_heterogeneo_estatico_bench.cu"
DEFAULT_SWEEP = [300, 400, 500, 600, 660, 700, 750, 800, 823, 850, 900, 950]

CSV_FIELDS = [
    "run_id", "order", "rep", "timestamp", "label",
    "precision", "NX", "NY", "ny_gpu", "ny_cpu", "steps", "warmup", "n_samples",
    "omp_threads",
    "mlups_total", "t_total_s",
    "t_step_ms_mean", "t_step_ms_std", "t_step_ms_median",
    "t_gpu_ms_mean", "t_gpu_ms_std", "t_gpu_ms_median",
    "t_cpu_ms_mean", "t_cpu_ms_std", "t_cpu_ms_median",
    "gpu_side_mlups", "cpu_side_mlups",
    "sample_ms", "gpu_temp_start", "gpu_temp_end", "gpu_temp_max", "gpu_sm_clock_mean", "gpu_power_max",
    "cpu_mhz_start", "cpu_mhz_end",
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
def build(precision):
    cmd = ["make", "-f", "Makefile.tfg", "build", f"N={SRC_NAME}"]
    if precision == "fp32":
        cmd.append("PRECISION=fp32")
    print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit("ERROR: fallo la compilacion (revisa que nvcc esta en el PATH).")
    suffix = "_fp32" if precision == "fp32" else ""
    return REPO / "bin" / (Path(SRC_NAME).stem + suffix)


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


def run_once(binary, ny_gpu, steps, warmup, threads, workdir, sample_ms=500):
    env = os.environ.copy()
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    cmd = [str(binary), "--ny-gpu", str(ny_gpu), "--steps", str(steps), "--warmup", str(warmup)]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--ny-gpu", type=int, nargs="+", default=DEFAULT_SWEEP,
                    help="filas asignadas a la GPU (el resto va a la CPU). Defecto: barrido 300..950")
    ap.add_argument("--reps", type=int, default=10, help="repeticiones por reparto (defecto 10)")
    ap.add_argument("--steps", type=int, default=1000, help="pasos LBM por ejecucion (defecto 1000)")
    ap.add_argument("--warmup", type=int, default=100, help="pasos iniciales descartados de las estadisticas")
    ap.add_argument("--threads", type=int, default=None, help="OMP_NUM_THREADS (defecto: todos los hilos)")
    ap.add_argument("--cooldown", type=float, default=5.0, help="segundos de pausa entre ejecuciones")
    ap.add_argument("--sample-ms", type=int, default=500,
                    help="cada cuantos ms se muestrea temperatura/reloj de la GPU durante la ejecucion "
                         "(0 = no muestrear; util para comprobar que el muestreo no perturba la medida)")
    ap.add_argument("--label", default="static", help="etiqueta para el nombre del CSV")
    ap.add_argument("--seed", type=int, default=12345, help="semilla para barajar el orden")
    ap.add_argument("--no-shuffle", action="store_true", help="no barajar el orden (no recomendado)")
    ap.add_argument("--no-warmup-run", action="store_true", help="saltar la ejecucion de calentamiento previa")
    ap.add_argument("--no-build", action="store_true", help="no recompilar")
    ap.add_argument("--binary", type=Path, default=None, help="usar este ejecutable (implica --no-build)")
    ap.add_argument("--outdir", type=Path, default=REPO / "results" / "benchmarks")
    args = ap.parse_args()

    if args.binary:
        binary = args.binary.resolve()
    elif args.no_build:
        suffix = "_fp32" if args.precision == "fp32" else ""
        binary = REPO / "bin" / (Path(SRC_NAME).stem + suffix)
    else:
        binary = build(args.precision)
    if not binary.exists():
        sys.exit(f"ERROR: no existe el ejecutable {binary}")

    workdir = REPO / "results" / "run"
    workdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.outdir / f"{stamp}_{args.label}_{args.precision}.csv"
    meta_path = csv_path.with_suffix(".json")

    commit, dirty = git_info()
    meta = machine_info()
    meta.update({"command": " ".join(sys.argv), "args": {k: str(v) for k, v in vars(args).items()},
                 "git_commit": commit, "git_dirty": dirty, "started": dt.datetime.now().isoformat(timespec="seconds")})
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    plan = [(ny, rep) for ny in args.ny_gpu for rep in range(1, args.reps + 1)]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(plan)
    total = len(plan)
    print(f"[plan] {total} ejecuciones ({len(args.ny_gpu)} repartos x {args.reps} repeticiones), "
          f"{args.steps} pasos cada una, precision {args.precision.upper()}")
    print(f"[plan] CSV: {csv_path}", flush=True)

    if not args.no_warmup_run:
        ny_w = sorted(args.ny_gpu)[len(args.ny_gpu) // 2]
        print(f"[warmup] ejecucion de calentamiento (ny_gpu={ny_w}), se descarta ...", flush=True)
        proc, res, _ = run_once(binary, ny_w, args.steps, args.warmup, args.threads, workdir, args.sample_ms)
        if res is None:
            sys.exit("ERROR: el programa no produjo BENCH_RESULT.\n--- stdout ---\n"
                     f"{proc.stdout}\n--- stderr ---\n{proc.stderr}")
        time.sleep(args.cooldown)

    t0 = time.time()
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, (ny, rep) in enumerate(plan, start=1):
            proc, res, extra = run_once(binary, ny, args.steps, args.warmup, args.threads, workdir, args.sample_ms)
            row = {k: "" for k in CSV_FIELDS}
            row.update({"run_id": f"{stamp}-{i:04d}", "order": i, "rep": rep,
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "label": args.label,
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
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(f"[{i:>3}/{total}] ny_gpu={ny:<4} rep={rep:<2} "
                  f"MLUPS={row['mlups_total']:<10} t_gpu={row['t_gpu_ms_mean']:<10} "
                  f"t_cpu={row['t_cpu_ms_mean']:<10} T_gpu={row['gpu_temp_max'] or '?'}C  "
                  f"(quedan ~{eta/60:.1f} min)", flush=True)
            if i < total:
                time.sleep(args.cooldown)

    print(f"\n[ok] datos guardados en {csv_path}")
    print(f"[ok] siguiente paso:  python3 scripts/analyze.py {csv_path}")


if __name__ == "__main__":
    main()
