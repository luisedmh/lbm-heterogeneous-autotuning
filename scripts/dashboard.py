#!/usr/bin/env python3
"""
dashboard.py - panel web (Streamlit) con TODAS las medidas guardadas en results/benchmarks/.

Pantalla principal: datos generales de todo lo medido, separados por precision.
Desde la barra lateral se filtra por fecha, precision, tamano de malla (columnas x filas),
obstaculo y GPU; la pestana "Comparar" pone lado a lado las medidas agrupadas por cualquiera
de esas variables.

Lanzar (desde la raiz del repo, en el PC de casa):
    streamlit run scripts/dashboard.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
y desde el portatil abrir un tunel SSH:
    ssh -N -L 8501:localhost:8501 luis@PC-Casa
para ver la pagina en  http://localhost:8501
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402

st.set_page_config(page_title="LBM CPU+GPU | Benchmarks", page_icon=":bar_chart:", layout="wide")

# ----------------------------------------------------------------------------------------
# Tema y paleta (categorica en orden fijo; el color sigue a la entidad, no a su posicion)
# ----------------------------------------------------------------------------------------
try:
    DARK = st.context.theme.type != "light"
except Exception:
    DARK = True

if DARK:
    INK, INK2, GRID, SURFACE = "#e8e7e1", "#b5b4aa", "rgba(255,255,255,0.09)", "rgba(0,0,0,0)"
    PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#2fa32f", "#9085e9", "#e66767"]
else:
    INK, INK2, GRID, SURFACE = "#1c1c1a", "#52514e", "rgba(0,0,0,0.08)", "rgba(0,0,0,0)"
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
C_GPU, C_CPU, C_TOT = PALETTE[0], PALETTE[1], INK2

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
      h1 {font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.1rem;}
      h2, h3 {font-weight: 600; letter-spacing: -0.005em;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-weight: 600;}
      [data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.75;}
      .subtle {opacity: 0.7; font-size: 0.9rem;}
    </style>
    """, unsafe_allow_html=True)

DASH_BY_PREC = {"FP64": "solid", "FP32": "dash"}


def style(fig, title=None, height=420, xtitle=None, ytitle=None, legend=True):
    """Estilo comun. El titulo se pinta con markdown encima del grafico (show) para que no choque con la leyenda."""
    fig.update_layout(
        meta=title, height=height, margin=dict(l=10, r=10, t=36 if legend else 12, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)) if legend else None,
        showlegend=legend, hovermode="x unified", hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(title=dict(text=xtitle, standoff=10), gridcolor=GRID, zeroline=False, linecolor=GRID,
                     ticks="outside", tickcolor=GRID, automargin=True)
    fig.update_yaxes(title=dict(text=ytitle, standoff=10), gridcolor=GRID, zeroline=False, linecolor=GRID,
                     automargin=True)
    return fig


def show(fig):
    if fig.layout.meta:
        st.markdown(f"##### {fig.layout.meta}")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}"


# ----------------------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo medidas ...")
def load_data(root, signature):
    return bd.load_all([Path(root)])


def signature_of(root):
    r = Path(root)
    if not r.exists():
        return ()
    return tuple((str(p), p.stat().st_mtime) for p in sorted(r.rglob("*.csv")))


st.sidebar.markdown("### Datos")
root = st.sidebar.text_input("Carpeta de medidas", str(bd.DEFAULT_ROOT), label_visibility="collapsed")
if st.sidebar.button("Recargar medidas", width="stretch"):
    st.cache_data.clear()
ALL = load_data(root, signature_of(root))

st.title("LBM D2Q9 heterogéneo CPU+GPU")
st.markdown('<div class="subtle">Velocidad real de cada dispositivo (MLUPS) medida dentro del programa '
            'heterogéneo de reparto estático</div>', unsafe_allow_html=True)

if ALL.empty:
    st.info(f"No hay medidas en `{root}`. Genera datos con:\n\n`python3 scripts/benchmark.py`")
    st.stop()

# ----------------------------------------------------------------------------------------
# Filtros (barra lateral)
# ----------------------------------------------------------------------------------------
st.sidebar.markdown("### Filtros")
dmin, dmax = dt.date.fromisoformat(ALL["date"].min()), dt.date.fromisoformat(ALL["date"].max())
if dmin == dmax:
    st.sidebar.caption(f"Fecha: {dmin.isoformat()} (único día medido)")
    d_from, d_to = dmin, dmax
else:
    rng = st.sidebar.date_input("Fecha", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    d_from, d_to = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax)

precs = sorted(ALL["precision"].unique(), reverse=True)           # FP64 antes que FP32
sel_prec = st.sidebar.multiselect("Precisión", precs, default=precs)
grids = list(ALL.drop_duplicates("grid").sort_values(["NX", "NY"])["grid"])
sel_grid = st.sidebar.multiselect("Tamaño de malla (columnas x filas)", grids, default=grids)
obsts = sorted(ALL["obstacle"].unique())
sel_obs = st.sidebar.multiselect("Obstáculo", obsts, default=obsts)
gpus = sorted(ALL["gpu_name"].unique())
sel_gpu = st.sidebar.multiselect("GPU", gpus, default=gpus) if len(gpus) > 1 else gpus

DF = ALL[(ALL["date"] >= d_from.isoformat()) & (ALL["date"] <= d_to.isoformat())
         & ALL["precision"].isin(sel_prec) & ALL["grid"].isin(sel_grid)
         & ALL["obstacle"].isin(sel_obs) & ALL["gpu_name"].isin(sel_gpu)].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(DF):,} de {len(ALL):,} ejecuciones seleccionadas")

if DF.empty:
    st.warning("Ninguna medida cumple los filtros elegidos. Amplía la selección en la barra lateral.")
    st.stop()

st.caption(f"{len(DF):,} ejecuciones · {DF['sweep_id'].nunique()} barridos · {DF['config'].nunique()} configuraciones · "
           f"{DF['date'].nunique()} día(s) ({DF['date'].min()} → {DF['date'].max()}) · "
           f"GPU: {', '.join(sorted(DF['gpu_name'].unique()))}")

tab_gen, tab_cmp, tab_det, tab_rep, tab_data = st.tabs(
    ["Resumen general", "Comparar", "Barrido en detalle", "Repetibilidad y temperatura", "Datos"])

# ========================================================================================
# 1) RESUMEN GENERAL: todo junto, separado por precision
# ========================================================================================
with tab_gen:
    st.subheader("Todas las medidas, por precisión")
    prec_present = [p for p in precs if p in set(DF["precision"])]
    cols = st.columns(len(prec_present))
    for col, p in zip(cols, prec_present):
        d = DF[DF["precision"] == p]
        by_cfg = bd.summarize(d, ["config", "ny_gpu"])
        best = by_cfg.loc[by_cfg["total_mlups"].idxmax()]
        with col.container(border=True):
            st.markdown(f"**{p}**")
            st.caption(f"{len(d):,} ejecuciones · {d['sweep_id'].nunique()} barridos · {d['grid'].nunique()} tamaño(s) · "
                       f"{d['obstacle'].nunique()} obstáculo(s)")
            m1, m2, m3 = st.columns(3)
            m1.metric("MLUPS GPU (media)", fmt(d["gpu_side_mlups"].mean()))
            m2.metric("MLUPS CPU (media)", fmt(d["cpu_side_mlups"].mean()))
            m3.metric("Mejor MLUPS total", fmt(best["total_mlups"]))
            st.caption(f"Mejor: {best['config']} con {int(best['ny_gpu'])} filas en la GPU")

    gen = bd.summarize(DF, ["precision", "alpha"])

    fig = go.Figure()
    for p in prec_present:
        g = gen[gen["precision"] == p].sort_values("alpha")
        dash = DASH_BY_PREC.get(p, "solid")
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["gpu_mlups"], mode="lines+markers", name=f"GPU · {p}",
                                 line=dict(color=C_GPU, width=2, dash=dash), marker=dict(size=7, line=dict(width=2, color="rgba(0,0,0,0)"))))
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["cpu_mlups"], mode="lines+markers", name=f"CPU · {p}",
                                 line=dict(color=C_CPU, width=2, dash=dash), marker=dict(size=7)))
    style(fig, "MLUPS de cada dispositivo según el reparto", 430,
          "fracción de filas asignadas a la GPU (α = filas GPU / filas totales)", "MLUPS de cada lado")
    show(fig)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for p in prec_present:
        g = gen[gen["precision"] == p].sort_values("alpha")
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["total_mlups"], mode="lines+markers", name=p,
                                 line=dict(color=C_TOT, width=2, dash=DASH_BY_PREC.get(p, "solid")), marker=dict(size=7)))
    style(fig, "MLUPS total del programa heterogéneo", 360, "α (fracción de filas GPU)", "MLUPS total")
    with c1:
        show(fig)
    fig = go.Figure()
    for p in prec_present:
        g = gen[gen["precision"] == p].sort_values("alpha")
        dash = DASH_BY_PREC.get(p, "solid")
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["t_gpu_ms"], mode="lines", name=f"GPU · {p}",
                                 line=dict(color=C_GPU, width=2, dash=dash)))
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["t_cpu_ms"], mode="lines", name=f"CPU · {p}",
                                 line=dict(color=C_CPU, width=2, dash=dash)))
    style(fig, "Tiempo por paso de cada lado", 360, "α (fracción de filas GPU)", "ms por paso")
    with c2:
        show(fig)
    st.caption("Cada punto es la media de todas las ejecuciones de la precisión con ese reparto. Si en el filtro hay "
               "mallas u obstáculos distintos, la media los mezcla: usa los filtros de la izquierda o la pestaña "
               "«Comparar» para verlos por separado.")

    st.subheader("Resumen por precisión y configuración")
    cfg = bd.summarize(DF, ["precision", "grid", "obstacle", "ny_gpu"])
    rows = []
    for (p, gr, ob), g in cfg.groupby(["precision", "grid", "obstacle"]):
        b = g.loc[g["total_mlups"].idxmax()]
        rows.append({"Precisión": p, "Malla": gr, "Obstáculo": ob,
                     "Ejecuciones": int(g["n"].sum()), "Repartos": len(g),
                     "MLUPS GPU (media)": g["gpu_mlups"].mean(), "MLUPS CPU (media)": g["cpu_mlups"].mean(),
                     "Mejor MLUPS total": b["total_mlups"], "Con filas GPU": int(b["ny_gpu"]),
                     "α del mejor": b["ny_gpu"] / int(gr.split("x")[1])})
    tb = pd.DataFrame(rows)
    st.dataframe(tb, hide_index=True, width="stretch", column_config={
        "MLUPS GPU (media)": st.column_config.NumberColumn(format="%.1f"),
        "MLUPS CPU (media)": st.column_config.NumberColumn(format="%.1f"),
        "Mejor MLUPS total": st.column_config.NumberColumn(format="%.1f"),
        "α del mejor": st.column_config.NumberColumn(format="%.3f")})

# ========================================================================================
# 2) COMPARAR: agrupar por cualquier variable
# ========================================================================================
GROUPS = {"Configuración completa": "config", "Precisión": "precision", "Tamaño de malla": "grid",
          "Obstáculo": "obstacle", "Fecha": "date", "Barrido": "sweep_id", "GPU": "gpu_name"}
METRICS = {"MLUPS de la GPU": ("gpu_mlups", "MLUPS"), "MLUPS de la CPU": ("cpu_mlups", "MLUPS"),
           "MLUPS total": ("total_mlups", "MLUPS"), "Tiempo GPU por paso": ("t_gpu_ms", "ms"),
           "Tiempo CPU por paso": ("t_cpu_ms", "ms"), "Tiempo del paso completo": ("t_step_ms", "ms")}
XAXES = {"Fracción de filas GPU (α)": ("alpha", "α = filas GPU / filas totales"),
         "Filas asignadas a la GPU": ("ny_gpu", "filas GPU")}

with tab_cmp:
    st.subheader("Comparar medidas")
    f1, f2, f3 = st.columns(3)
    g_label = f1.selectbox("Agrupar por", list(GROUPS), index=0)
    m_label = f2.selectbox("Métrica", list(METRICS), index=0)
    x_label = f3.selectbox("Eje horizontal", list(XAXES), index=0)
    gcol, (mcol, munit), (xcol, xtitle) = GROUPS[g_label], METRICS[m_label], XAXES[x_label]

    if xcol == "ny_gpu" and DF["NY"].nunique() > 1:
        st.info("Hay mallas con distinto número de filas: con «Filas asignadas a la GPU» no son comparables entre "
                "sí. Usa la fracción α o filtra por un tamaño.")

    # el color sigue a la entidad: se asigna sobre TODAS las medidas, no solo las filtradas
    ordered_all = sorted(ALL[gcol].astype(str).unique())
    color_of = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(ordered_all)}
    counts = DF.groupby(gcol).size().sort_values(ascending=False)
    shown = list(counts.index[:8])
    if len(counts) > 8:
        st.warning(f"Hay {len(counts)} grupos; se muestran los 8 con más ejecuciones. Filtra para ver el resto.")
    DS = DF[DF[gcol].isin(shown)]
    cmp_df = bd.summarize(DS, [gcol, xcol])

    fig = go.Figure()
    for v in sorted(shown, key=str):
        g = cmp_df[cmp_df[gcol] == v].sort_values(xcol)
        fig.add_trace(go.Scatter(x=g[xcol], y=g[mcol], mode="lines+markers", name=str(v),
                                 line=dict(color=color_of[str(v)], width=2), marker=dict(size=7)))
    style(fig, f"{m_label} según el reparto, por {g_label.lower()}", 470, xtitle, munit)
    show(fig)
    st.caption("Cada línea es la media de las ejecuciones de ese grupo en cada reparto. Si no agrupas por una variable "
               "que está mezclada en los datos (p. ej. agrupas por precisión con dos obstáculos), cada punto promedia "
               "esos casos: filtra a la izquierda para aislarla.")

    st.subheader(f"Mejor rendimiento total por {g_label.lower()}")
    best_rows = []
    for v in sorted(shown, key=str):
        s = bd.summarize(DS[DS[gcol] == v], ["ny_gpu"])
        b = s.loc[s["total_mlups"].idxmax()]
        sub = DS[DS[gcol] == v]
        best_rows.append({g_label: str(v), "Ejecuciones": len(sub), "MLUPS GPU (media)": sub["gpu_side_mlups"].mean(),
                          "MLUPS CPU (media)": sub["cpu_side_mlups"].mean(), "Mejor MLUPS total": b["total_mlups"],
                          "Con filas GPU": int(b["ny_gpu"]), "NY": int(sub["NY"].median()),
                          "α del mejor": b["ny_gpu"] / float(sub["NY"].median())})
    bt = pd.DataFrame(best_rows)
    bar = go.Figure()
    bar.add_trace(go.Bar(y=bt[g_label], x=bt["Mejor MLUPS total"], orientation="h",
                         marker=dict(color=[color_of[v] for v in bt[g_label]], cornerradius=4),
                         text=[f"{x:,.0f}" for x in bt["Mejor MLUPS total"]], textposition="outside",
                         textfont=dict(color=INK), hovertemplate="%{y}: %{x:,.1f} MLUPS<extra></extra>"))
    style(bar, None, max(220, 52 * len(bt) + 60), "MLUPS total (mejor reparto de cada grupo)", None, legend=False)
    bar.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    bar.update_layout(hovermode="closest", bargap=0.35)
    show(bar)
    st.dataframe(bt.drop(columns=["NY"]), hide_index=True, width="stretch", column_config={
        "MLUPS GPU (media)": st.column_config.NumberColumn(format="%.1f"),
        "MLUPS CPU (media)": st.column_config.NumberColumn(format="%.1f"),
        "Mejor MLUPS total": st.column_config.NumberColumn(format="%.1f"),
        "α del mejor": st.column_config.NumberColumn(format="%.3f")})

# ========================================================================================
# 3) BARRIDO EN DETALLE (un CSV)
# ========================================================================================
sweeps = (DF.groupby("sweep_id").agg(datetime=("datetime", "min"), n=("order", "size"), config=("config", "first"))
          .sort_values("datetime", ascending=False))
with tab_det:
    st.subheader("Un barrido en detalle")
    sweep = st.selectbox("Barrido", list(sweeps.index),
                         format_func=lambda s: f"{s}   ({sweeps.loc[s, 'n']} ejecuciones)")
    S = DF[DF["sweep_id"] == sweep].copy()
    SS = bd.summarize(S, ["ny_gpu"]).sort_values("ny_gpu")
    r0 = S.iloc[0]
    st.caption(f"{r0['precision']} · malla {r0['grid']} (columnas x filas) · obstáculo {r0['obstacle']} · {r0['date']} · "
               f"{r0['steps']:.0f} pasos ({r0['warmup']:.0f} de calentamiento) · {r0['omp_threads']:.0f} hilos de CPU")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MLUPS GPU", f"{SS['gpu_mlups'].min():.0f} – {SS['gpu_mlups'].max():.0f}", border=True)
    k2.metric("MLUPS CPU", f"{SS['cpu_mlups'].min():.0f} – {SS['cpu_mlups'].max():.0f}", border=True)
    bs = SS.loc[SS["total_mlups"].idxmax()]
    k3.metric("Mejor MLUPS total", fmt(bs["total_mlups"]), f"{int(bs['ny_gpu'])} filas GPU", delta_color="off",
              delta_arrow="off", border=True)
    k4.metric("Repeticiones por reparto", f"{int(SS['n'].min())}", border=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S["ny_gpu"], y=S["gpu_side_mlups"], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(color=C_GPU, opacity=0.25, size=6)))
    fig.add_trace(go.Scatter(x=S["ny_gpu"], y=S["cpu_side_mlups"], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(color=C_CPU, opacity=0.25, size=6)))
    fig.add_trace(go.Scatter(x=SS["ny_gpu"], y=SS["gpu_mlups"], mode="lines+markers", name="GPU (media)",
                             line=dict(color=C_GPU, width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=SS["ny_gpu"], y=SS["cpu_mlups"], mode="lines+markers", name="CPU (media)",
                             line=dict(color=C_CPU, width=2), marker=dict(size=8)))
    style(fig, "MLUPS de la GPU y de la CPU", 430, "filas asignadas a la GPU (la CPU hace el resto)", "MLUPS de cada lado")
    show(fig)
    st.caption("MLUPS de un lado = celdas de ese lado / tiempo que ese lado tarda en un paso (GPU: cudaEvent, incluye la "
               "copia del halo; CPU: reloj alrededor de su cálculo). Puntos claros = cada ejecución; línea = media.")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=SS["ny_gpu"], y=SS["t_gpu_ms"], mode="lines+markers", name="GPU", line=dict(color=C_GPU, width=2)))
    fig.add_trace(go.Scatter(x=SS["ny_gpu"], y=SS["t_cpu_ms"], mode="lines+markers", name="CPU", line=dict(color=C_CPU, width=2)))
    style(fig, "Tiempo por paso de cada lado", 360, "filas GPU", "ms por paso")
    with c1:
        show(fig)
    fig = go.Figure(go.Scatter(x=SS["ny_gpu"], y=SS["total_mlups"], mode="lines+markers", name="MLUPS total",
                               error_y=dict(type="data", array=SS["total_std"].fillna(0), visible=True, color=INK2),
                               line=dict(color=C_TOT, width=2), marker=dict(size=7)))
    style(fig, "MLUPS total del programa heterogéneo", 360, "filas GPU", "MLUPS", legend=False)
    with c2:
        show(fig)

    tb = SS[["ny_gpu", "n", "gpu_mlups", "cpu_mlups", "total_mlups", "total_cv_pct", "t_gpu_ms", "t_cpu_ms", "t_step_ms"]].copy()
    tb.insert(1, "ny_cpu", int(r0["NY"]) - tb["ny_gpu"])
    tb = tb.rename(columns={"ny_gpu": "filas GPU", "ny_cpu": "filas CPU", "n": "reps", "gpu_mlups": "MLUPS GPU",
                            "cpu_mlups": "MLUPS CPU", "total_mlups": "MLUPS total", "total_cv_pct": "CV %",
                            "t_gpu_ms": "t GPU (ms)", "t_cpu_ms": "t CPU (ms)", "t_step_ms": "t paso (ms)"})
    st.dataframe(tb, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tb.columns
                                if c not in ("filas GPU", "filas CPU", "reps")})

# ========================================================================================
# 4) REPETIBILIDAD Y TEMPERATURA (del barrido elegido)
# ========================================================================================
with tab_rep:
    st.subheader("Repetibilidad y temperatura")
    st.caption(f"Barrido: {sweep}")
    c1, c2 = st.columns(2)
    for col, key, title, color in ((c1, "gpu_side_mlups", "MLUPS de la GPU por reparto", C_GPU),
                                   (c2, "cpu_side_mlups", "MLUPS de la CPU por reparto", C_CPU)):
        fb = go.Figure()
        for ny in sorted(S["ny_gpu"].unique()):
            fb.add_trace(go.Box(y=S[S["ny_gpu"] == ny][key], name=str(int(ny)), boxpoints="all", jitter=0.4, pointpos=0,
                                marker=dict(size=4, color=color), line=dict(color=color, width=1.5),
                                fillcolor="rgba(0,0,0,0)"))
        style(fb, title, 380, "filas GPU", "MLUPS", legend=False)
        fb.update_layout(hovermode="closest")
        with col:
            show(fb)
    st.caption("El orden de las ejecuciones está barajado a propósito: si hay deriva térmica se ve aquí sin confundirse "
               "con el efecto del reparto.")
    c1, c2 = st.columns(2)
    for col, key, title in ((c1, "gpu_side_mlups", "MLUPS GPU según el orden de ejecución"),
                            (c2, "cpu_side_mlups", "MLUPS CPU según el orden de ejecución")):
        fo = go.Figure(go.Scatter(x=S["order"], y=S[key], mode="markers",
                                  marker=dict(color=S["ny_gpu"], colorscale="Viridis", showscale=True,
                                              colorbar=dict(title="filas GPU", thickness=10), size=7)))
        style(fo, title, 340, "nº de ejecución", "MLUPS", legend=False)
        fo.update_layout(hovermode="closest")
        with col:
            show(fo)
    if S["gpu_temp_max"].notna().any():
        c1, c2 = st.columns(2)
        ft = go.Figure(go.Scatter(x=S["order"], y=S["gpu_temp_max"], mode="lines+markers",
                                  line=dict(color=C_CPU, width=2), marker=dict(size=5)))
        style(ft, "Temperatura máxima de la GPU en cada ejecución", 320, "nº de ejecución", "°C", legend=False)
        with c1:
            show(ft)
        if S["gpu_sm_clock_mean"].notna().any():
            fc = go.Figure(go.Scatter(x=S["order"], y=S["gpu_sm_clock_mean"], mode="lines+markers",
                                      line=dict(color=C_GPU, width=2), marker=dict(size=5)))
            style(fc, "Reloj medio de la GPU", 320, "nº de ejecución", "MHz", legend=False)
            with c2:
                show(fc)
    else:
        st.info("Este barrido no tiene datos de temperatura (nvidia-smi no estaba disponible al medir).")

# ========================================================================================
# 5) DATOS
# ========================================================================================
with tab_data:
    st.subheader("Barridos incluidos")
    inv = (DF.groupby("sweep_id").agg(Fecha=("date", "first"), Precisión=("precision", "first"), Malla=("grid", "first"),
                                      Obstáculo=("obstacle", "first"), Ejecuciones=("order", "size"),
                                      Repartos=("ny_gpu", "nunique"), GPU=("gpu_name", "first"),
                                      Archivo=("source_file", "first"))
           .reset_index().sort_values("Fecha", ascending=False).rename(columns={"sweep_id": "Barrido"}))
    st.dataframe(inv, hide_index=True, width="stretch")

    st.subheader("Ejecuciones individuales (filtradas)")
    st.dataframe(DF.drop(columns=["datetime"]), hide_index=True, width="stretch")
    st.download_button("Descargar las ejecuciones filtradas (CSV)", DF.drop(columns=["datetime"]).to_csv(index=False).encode(),
                       file_name="ejecuciones_filtradas.csv", mime="text/csv")

    st.subheader("Máquina y configuración del barrido elegido")
    js = bd.sidecar(Path(bd.REPO) / S.iloc[0]["source_file"]) if "source_file" in S.columns else {}
    if js:
        st.json(js, expanded=False)
    else:
        st.caption("Este barrido no tiene fichero .json junto al CSV.")
