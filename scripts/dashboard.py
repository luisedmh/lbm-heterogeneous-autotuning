#!/usr/bin/env python3
"""
dashboard.py - interfaz web (Streamlit) para ver los MLUPS de GPU y de CPU medidos
con scripts/benchmark.py (heterogeneo estatico).

Lanzar (desde la raiz del repo, en el PC de casa):
    streamlit run scripts/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
y desde el portatil abrir un tunel SSH:
    ssh -N -L 8501:localhost:8501 luis@PC-Casa
para ver la pagina en  http://localhost:8501
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402  (mismo directorio)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO / "results" / "benchmarks"

st.set_page_config(page_title="LBM CPU+GPU - MLUPS", layout="wide")

C_GPU, C_CPU, C_STEP = "#2a9d8f", "#e76f51", "#264653"


@st.cache_data(show_spinner=False)
def load_files(paths, mtimes):
    return analyze.load([Path(p) for p in paths])


def is_sweep(path):
    """True si es un CSV de benchmark.py (tiene la columna ny_gpu); ignora summary_*.csv."""
    if Path(path).name.startswith(("summary_", "model_", "fit_")):
        return False
    try:
        return "ny_gpu" in set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False


# --- barra lateral: que datos mirar -------------------------------------------------
st.sidebar.title("Datos")
data_dir = Path(st.sidebar.text_input("Carpeta de resultados", str(DEFAULT_DIR)))
csvs = sorted([p for p in data_dir.glob("*.csv") if is_sweep(p)], reverse=True)
if not csvs:
    st.title("LBM CPU+GPU - MLUPS")
    st.info(f"No hay CSV de benchmark en `{data_dir}`. Genera datos primero:\n\n"
            "`python3 scripts/benchmark.py --precision fp64 --reps 10 --label static`")
    st.stop()

sel = st.sidebar.multiselect("Ficheros de benchmark", [p.name for p in csvs], default=[csvs[0].name])
if not sel:
    st.warning("Selecciona al menos un fichero en la barra lateral.")
    st.stop()
df = load_files(tuple(str(data_dir / n) for n in sel), tuple((data_dir / n).stat().st_mtime for n in sel))
if df.empty:
    st.error("Los ficheros elegidos no tienen ejecuciones validas.")
    st.stop()

prec = st.sidebar.selectbox("Precision", sorted(df["precision"].unique()), index=0)
d = df[df["precision"] == prec].copy()
NX, NY = int(d["NX"].iloc[0]), int(d["NY"].iloc[0])
summ = analyze.summarize(df)
summ = summ[summ["precision"] == prec].reset_index(drop=True)

meta_path = (data_dir / sel[0]).with_suffix(".json")
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

# --- cabecera -------------------------------------------------------------------------
st.title("LBM D2Q9 heterogeneo CPU+GPU - MLUPS de GPU y de CPU")
sub = f"{prec} - {NX}x{NY} - {len(d)} ejecuciones - {d['ny_gpu'].nunique()} repartos"
if meta:
    sub += f" - GPU: {meta.get('gpu_name')} - CPU: {meta.get('cpu_model')}"
st.caption(sub)

k1, k2, k3 = st.columns(3)
k1.metric("MLUPS GPU (rango entre repartos)", f"{summ['gpu_side_mlups'].min():.0f} - {summ['gpu_side_mlups'].max():.0f}")
k2.metric("MLUPS CPU (rango entre repartos)", f"{summ['cpu_side_mlups'].min():.0f} - {summ['cpu_side_mlups'].max():.0f}")
k3.metric("Repeticiones por reparto", f"{int(summ['n'].min())}")

tab_m, tab_rep, tab_data = st.tabs(["MLUPS de GPU y de CPU", "Repetibilidad y temperatura", "Datos"])

# --- MLUPS de cada lado -----------------------------------------------------------------
with tab_m:
    st.subheader("Velocidad de cada dispositivo, medida con CPU y GPU trabajando a la vez")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["ny_gpu"], y=d["gpu_side_mlups"], mode="markers", showlegend=False,
                             marker=dict(color=C_GPU, opacity=0.25, size=6)))
    fig.add_trace(go.Scatter(x=d["ny_gpu"], y=d["cpu_side_mlups"], mode="markers", showlegend=False,
                             marker=dict(color=C_CPU, opacity=0.25, size=6)))
    fig.add_trace(go.Scatter(x=summ["ny_gpu"], y=summ["gpu_side_mlups"], mode="lines+markers", name="GPU (media)",
                             line=dict(color=C_GPU, width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=summ["ny_gpu"], y=summ["cpu_side_mlups"], mode="lines+markers", name="CPU (media)",
                             line=dict(color=C_CPU, width=3), marker=dict(size=9)))
    fig.update_layout(xaxis_title="filas asignadas a la GPU (la CPU hace el resto)", yaxis_title="MLUPS de cada lado",
                      height=460, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("MLUPS de un lado = celdas de ese lado / tiempo que ese lado tarda en un paso (GPU: cudaEvent, incluye "
               "la copia del halo; CPU: reloj alrededor de su calculo). Puntos claros = cada ejecucion; linea = media.")

    c1, c2 = st.columns(2)
    f1 = go.Figure(go.Scatter(x=summ["ny_gpu"], y=summ["t_gpu_ms"], mode="lines+markers", name="GPU",
                              line=dict(color=C_GPU, width=3)))
    f1.add_trace(go.Scatter(x=summ["ny_gpu"], y=summ["t_cpu_ms"], mode="lines+markers", name="CPU",
                            line=dict(color=C_CPU, width=3)))
    f1.update_layout(title="Tiempo por paso de cada lado", xaxis_title="filas GPU", yaxis_title="ms por paso",
                     height=380, legend=dict(orientation="h", y=1.15))
    c1.plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Scatter(x=summ["ny_gpu"], y=summ["mlups_mean"], mode="lines+markers",
                              error_y=dict(type="data", array=summ["mlups_std"].fillna(0), visible=True),
                              line=dict(color=C_STEP, width=3)))
    f2.update_layout(title="MLUPS total del programa heterogeneo", xaxis_title="filas GPU", yaxis_title="MLUPS",
                     height=380)
    c2.plotly_chart(f2, width="stretch")

    st.subheader("Tabla por reparto (media de las repeticiones)")
    tb = summ[["ny_gpu", "ny_cpu", "n", "gpu_side_mlups", "cpu_side_mlups", "mlups_mean", "mlups_cv_pct",
               "t_gpu_ms", "t_cpu_ms", "t_step_ms"]].rename(columns={
        "ny_gpu": "filas GPU", "ny_cpu": "filas CPU", "n": "reps", "gpu_side_mlups": "MLUPS GPU",
        "cpu_side_mlups": "MLUPS CPU", "mlups_mean": "MLUPS total", "mlups_cv_pct": "CV %",
        "t_gpu_ms": "t GPU (ms)", "t_cpu_ms": "t CPU (ms)", "t_step_ms": "t paso (ms)"})
    st.dataframe(tb, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tb.columns
                                if c not in ("filas GPU", "filas CPU", "reps")})

# --- repetibilidad -----------------------------------------------------------------------
with tab_rep:
    st.subheader("Variabilidad entre repeticiones")
    c1, c2 = st.columns(2)
    for col, key, title, color in ((c1, "gpu_side_mlups", "MLUPS de la GPU", C_GPU),
                                   (c2, "cpu_side_mlups", "MLUPS de la CPU", C_CPU)):
        fb = go.Figure()
        for ny in sorted(d["ny_gpu"].unique()):
            fb.add_trace(go.Box(y=d[d["ny_gpu"] == ny][key], name=str(int(ny)), boxpoints="all", jitter=0.4,
                                pointpos=0, marker=dict(size=5, color=color), line=dict(color=color)))
        fb.update_layout(title=title, xaxis_title="filas GPU", yaxis_title="MLUPS", height=400, showlegend=False)
        col.plotly_chart(fb, width="stretch")

    st.subheader("Orden de ejecucion y temperatura")
    st.caption("El orden de las ejecuciones esta barajado a proposito: si hay deriva termica se ve aqui sin "
               "confundirse con el efecto del reparto.")
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "gpu_side_mlups", "MLUPS GPU segun el orden de ejecucion"),
                            (c2, "cpu_side_mlups", "MLUPS CPU segun el orden de ejecucion")):
        fo = go.Figure(go.Scatter(x=d["order"], y=d[key], mode="markers",
                                  marker=dict(color=d["ny_gpu"], colorscale="Viridis", showscale=True,
                                              colorbar=dict(title="filas GPU"), size=8)))
        fo.update_layout(title=title, xaxis_title="n de ejecucion", yaxis_title="MLUPS", height=380)
        col.plotly_chart(fo, width="stretch")
    if d["gpu_temp_max"].notna().any():
        ft = go.Figure(go.Scatter(x=d["order"], y=d["gpu_temp_max"], mode="lines+markers", line=dict(color=C_CPU)))
        ft.update_layout(title="Temperatura maxima de la GPU en cada ejecucion", xaxis_title="n de ejecucion",
                         yaxis_title="T max GPU (C)", height=340)
        st.plotly_chart(ft, width="stretch")
        if d["gpu_sm_clock_mean"].notna().any():
            fc = go.Figure(go.Scatter(x=d["order"], y=d["gpu_sm_clock_mean"], mode="lines+markers",
                                      line=dict(color=C_GPU)))
            fc.update_layout(title="Reloj medio de la GPU (MHz)", xaxis_title="n de ejecucion", yaxis_title="MHz",
                             height=340)
            st.plotly_chart(fc, width="stretch")
    else:
        st.info("Sin datos de temperatura (nvidia-smi no estaba disponible al medir).")

# --- datos ---------------------------------------------------------------------------------
with tab_data:
    st.subheader("Ejecuciones individuales")
    st.dataframe(d.sort_values("order"), hide_index=True, width="stretch")
    st.download_button("Descargar estas ejecuciones (CSV)", d.to_csv(index=False).encode(),
                       file_name=f"ejecuciones_{prec}.csv", mime="text/csv")
    if meta:
        st.subheader("Maquina y configuracion de la medida")
        st.json(meta)
