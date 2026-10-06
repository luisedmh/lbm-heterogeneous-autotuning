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


def style(fig, title=None, height=420, xtitle=None, ytitle=None, legend=True, xrange=None):
    """Estilo comun. El titulo se pinta con markdown encima del grafico (show) para que no choque con la leyenda."""
    fig.update_layout(
        meta=title, height=height, margin=dict(l=10, r=10, t=36 if legend else 12, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=INK2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)) if legend else None,
        showlegend=legend, hovermode="x unified", hoverlabel=dict(font_size=12),
    )
    # Ejes SIN zoom: la X abarca todo el dominio posible (0..1 o 0..filas) y la Y arranca en 0, para que una
    # variacion pequena no parezca enorme y todos los graficos se lean con la misma escala.
    fig.update_xaxes(title=dict(text=xtitle, standoff=10), gridcolor=GRID, zeroline=False, linecolor=GRID,
                     ticks="outside", tickcolor=GRID, automargin=True, range=xrange)
    fig.update_yaxes(title=dict(text=ytitle, standoff=10), gridcolor=GRID, zeroline=False, linecolor=GRID,
                     automargin=True, rangemode="tozero")
    return fig


def show(fig):
    if fig.layout.meta:
        st.markdown(f"##### {fig.layout.meta}")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def fmt(v, nd=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{nd}f}"


def rng_txt(series):
    return f"{series.min():.0f} – {series.max():.0f}"


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
steps_all = sorted(ALL["steps"].dropna().astype(int).unique())
steps_default = [x for x in steps_all if x >= 1000] or steps_all      # las pruebas cortas no cuentan por defecto
sel_steps = st.sidebar.multiselect("Pasos por ejecución", steps_all, default=steps_default,
                                   help="Menos de 1000 pasos es una prueba rápida y la medida es poco fiable, por eso "
                                        "se ocultan por defecto. Añádelas aquí si quieres verlas.")
if set(steps_default) != set(steps_all):
    st.sidebar.caption("Las pruebas de menos de 1000 pasos están ocultas (se pueden añadir en este filtro).")
gpus = sorted(ALL["gpu_name"].unique())
sel_gpu = st.sidebar.multiselect("GPU", gpus, default=gpus) if len(gpus) > 1 else gpus

DF = ALL[(ALL["date"] >= d_from.isoformat()) & (ALL["date"] <= d_to.isoformat())
         & ALL["precision"].isin(sel_prec) & ALL["grid"].isin(sel_grid) & ALL["obstacle"].isin(sel_obs)
         & ALL["steps"].astype("Int64").isin(sel_steps) & ALL["gpu_name"].isin(sel_gpu)].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(DF):,} de {len(ALL):,} ejecuciones seleccionadas")

if DF.empty:
    st.warning("Ninguna medida cumple los filtros elegidos. Amplía la selección en la barra lateral.")
    st.stop()

# Barrido a inspeccionar (pestanas "Barrido en detalle" y "Calidad"): por defecto el MAS COMPLETO
sweeps = (DF.groupby("sweep_id").agg(datetime=("datetime", "min"), n=("order", "size"), splits=("ny_gpu", "nunique"))
          .sort_values(["splits", "n", "datetime"], ascending=False))
st.sidebar.markdown("### Barrido a inspeccionar")
sweep = st.sidebar.selectbox("Barrido", list(sweeps.index), label_visibility="collapsed",
                             format_func=lambda s: f"{s}  ({sweeps.loc[s, 'splits']} repartos, {sweeps.loc[s, 'n']} ejec.)")
st.sidebar.caption("Se usa en «Barrido en detalle» y «Calidad de la medida». Los más completos salen primero.")

st.caption(f"{len(DF):,} ejecuciones · {DF['sweep_id'].nunique()} barridos · {DF['config'].nunique()} configuraciones · "
           f"{DF['date'].nunique()} día(s) ({DF['date'].min()} → {DF['date'].max()}) · "
           f"GPU: {', '.join(sorted(DF['gpu_name'].unique()))}")

tab_gen, tab_cmp, tab_det, tab_rep, tab_data = st.tabs(
    ["Resumen general", "Comparar", "Barrido en detalle", "Calidad de la medida", "Datos"])


def pooled(d, keys):
    """Media por `keys` + nº de configuraciones distintas que entran en cada punto."""
    s = bd.summarize(d, keys)
    k = d.groupby(list(keys))["config"].nunique().rename("n_cfg").reset_index()
    return s.merge(k, on=list(keys))


# ========================================================================================
# 1) RESUMEN GENERAL: todo junto, separado por precision
# ========================================================================================
with tab_gen:
    st.subheader("Todas las medidas, por precisión")
    prec_present = [p for p in precs if p in set(DF["precision"])]

    mixed = [p for p in prec_present if DF[DF["precision"] == p]["config"].nunique() > 1]
    if mixed:
        st.info("En " + " y ".join(mixed) + " hay varias configuraciones (mallas u obstáculos distintos). Las curvas "
                "gruesas son la **media de todas**; las finas, cada configuración. Para ver una sola, "
                "filtra en la barra lateral.")
    if DF["steps"].nunique() > 1:
        st.warning("Hay medidas con distinto número de pasos (" + ", ".join(str(int(x)) for x in sorted(DF['steps'].unique()))
                   + "). Las de pocos pasos son menos fiables: filtra por «Pasos por ejecución» si no quieres mezclarlas.")

    cols = st.columns(len(prec_present))
    for col, p in zip(cols, prec_present):
        d = DF[DF["precision"] == p]
        by_cfg = bd.summarize(d, ["config", "ny_gpu"])
        best = by_cfg.loc[by_cfg["total_mlups"].idxmax()]
        by_split = bd.summarize(d, ["alpha"])
        with col.container(border=True):
            st.markdown(f"**{p}**")
            st.caption(f"{len(d):,} ejecuciones · {d['sweep_id'].nunique()} barridos · {d['grid'].nunique()} tamaño(s) · "
                       f"{d['obstacle'].nunique()} obstáculo(s)")
            m1, m2, m3 = st.columns(3)
            m1.metric("MLUPS GPU (rango)", rng_txt(by_split["gpu_mlups"]), help="Mínimo y máximo de la media por reparto")
            m2.metric("MLUPS CPU (rango)", rng_txt(by_split["cpu_mlups"]), help="Mínimo y máximo de la media por reparto")
            m3.metric("Mejor MLUPS total", fmt(best["total_mlups"]))
            st.caption(f"Mejor: {best['config']} con {int(best['ny_gpu'])} filas en la GPU")

    fig = go.Figure()
    for p in prec_present:
        d = DF[DF["precision"] == p]
        dash = DASH_BY_PREC.get(p, "solid")
        if d["config"].nunique() > 1:                       # curvas finas: cada configuracion
            for cfg_name, dc in d.groupby("config"):
                sc = bd.summarize(dc, ["alpha"]).sort_values("alpha")
                for ycol, color in (("gpu_mlups", C_GPU), ("cpu_mlups", C_CPU)):
                    fig.add_trace(go.Scatter(x=sc["alpha"], y=sc[ycol], mode="lines", showlegend=False, hoverinfo="skip",
                                             line=dict(color=color, width=1.5, dash=dash), opacity=0.45))
        g = pooled(d, ["alpha"]).sort_values("alpha")
        for ycol, name, color in (("gpu_mlups", "GPU", C_GPU), ("cpu_mlups", "CPU", C_CPU)):
            fig.add_trace(go.Scatter(x=g["alpha"], y=g[ycol], mode="lines+markers", name=f"{name} · {p}",
                                     customdata=np.stack([g["n"], g["n_cfg"]], axis=-1),
                                     hovertemplate="%{y:,.1f}  (media de %{customdata[0]} ejec., %{customdata[1]} config.)<extra>" + f"{name} · {p}" + "</extra>",
                                     line=dict(color=color, width=2.5, dash=dash), marker=dict(size=7)))
    style(fig, "MLUPS de cada dispositivo según el reparto", 430,
          "fracción de filas asignadas a la GPU (α = filas GPU / filas totales)", "MLUPS de cada lado")
    show(fig)

    fig = go.Figure()
    for p in prec_present:
        d = DF[DF["precision"] == p]
        dash = DASH_BY_PREC.get(p, "solid")
        if d["config"].nunique() > 1:
            for cfg_name, dc in d.groupby("config"):
                sc = bd.summarize(dc, ["alpha"]).sort_values("alpha")
                fig.add_trace(go.Scatter(x=sc["alpha"], y=sc["total_mlups"], mode="lines", showlegend=False, hoverinfo="skip",
                                         line=dict(color=C_TOT, width=1.5, dash=dash), opacity=0.45))
        g = pooled(d, ["alpha"]).sort_values("alpha")
        fig.add_trace(go.Scatter(x=g["alpha"], y=g["total_mlups"], mode="lines+markers", name=p,
                                 customdata=np.stack([g["n"], g["n_cfg"]], axis=-1),
                                 hovertemplate="%{y:,.1f}  (media de %{customdata[0]} ejec., %{customdata[1]} config.)<extra>" + p + "</extra>",
                                 line=dict(color=C_TOT, width=2.5, dash=dash), marker=dict(size=7)))
    style(fig, "MLUPS total del programa heterogéneo (CPU + GPU)", 380, "α (fracción de filas GPU)", "MLUPS total")
    show(fig)

    fig = go.Figure()
    for p in prec_present:
        d = DF[DF["precision"] == p]
        dash = DASH_BY_PREC.get(p, "solid")
        if d["config"].nunique() > 1:
            for cfg_name, dc in d.groupby("config"):
                sc = bd.summarize(dc, ["alpha"]).sort_values("alpha")
                for ycol, color in (("t_gpu_ms", C_GPU), ("t_cpu_ms", C_CPU)):
                    fig.add_trace(go.Scatter(x=sc["alpha"], y=sc[ycol], mode="lines", showlegend=False, hoverinfo="skip",
                                             line=dict(color=color, width=1.5, dash=dash), opacity=0.45))
        g = pooled(d, ["alpha"]).sort_values("alpha")
        for ycol, name, color in (("t_gpu_ms", "GPU", C_GPU), ("t_cpu_ms", "CPU", C_CPU)):
            fig.add_trace(go.Scatter(x=g["alpha"], y=g[ycol], mode="lines+markers", name=f"{name} · {p}",
                                     customdata=np.stack([g["n"], g["n_cfg"]], axis=-1),
                                     hovertemplate="%{y:,.3f} ms  (media de %{customdata[0]} ejec., %{customdata[1]} config.)<extra>" + f"{name} · {p}" + "</extra>",
                                     line=dict(color=color, width=2.5, dash=dash), marker=dict(size=7)))
    style(fig, "Tiempo por paso de cada lado (t GPU y t CPU)", 400, "α (fracción de filas GPU)", "ms por paso")
    show(fig)
    st.caption("El paso completo dura lo que tarde el lado más lento: donde se cruzan las dos curvas, GPU y CPU tardan lo "
               "mismo y ninguna espera a la otra.")

    st.subheader("Resumen por configuración")
    cfg = bd.summarize(DF, ["precision", "grid", "obstacle", "ny_gpu"])
    rows = []
    for (p, gr, ob), g in cfg.groupby(["precision", "grid", "obstacle"]):
        b = g.loc[g["total_mlups"].idxmax()]
        rows.append({"Precisión": p, "Malla": gr, "Obstáculo": ob,
                     "Ejecuciones": int(g["n"].sum()), "Repartos": len(g),
                     "MLUPS GPU": rng_txt(g["gpu_mlups"]), "MLUPS CPU": rng_txt(g["cpu_mlups"]),
                     "Mejor MLUPS total": b["total_mlups"], "Con filas GPU": int(b["ny_gpu"]),
                     "α del mejor": b["ny_gpu"] / int(gr.split("x")[1])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Mejor MLUPS total": st.column_config.NumberColumn(format="%.1f"),
        "α del mejor": st.column_config.NumberColumn(format="%.3f")})
    st.caption("MLUPS GPU / CPU: mínimo – máximo de la media por reparto, medidos dentro del programa heterogéneo.")

# ========================================================================================
# 2) COMPARAR: agrupar por cualquier variable
# ========================================================================================
GROUPS = {"Configuración completa": "config", "Precisión": "precision", "Tamaño de malla": "grid",
          "Obstáculo": "obstacle", "Fecha": "date", "Barrido": "sweep_id", "GPU": "gpu_name"}
METRICS = {"MLUPS de la GPU": ("gpu_mlups", "MLUPS"), "MLUPS de la CPU": ("cpu_mlups", "MLUPS"),
           "MLUPS total": ("total_mlups", "MLUPS"), "Tiempo GPU por paso": ("t_gpu_ms", "ms"),
           "Tiempo CPU por paso": ("t_cpu_ms", "ms")}
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
                                 line=dict(color=color_of[str(v)], width=2.5), marker=dict(size=7)))
    xr = [0, 1] if xcol == "alpha" else [0, int(DS["NY"].max())]
    style(fig, f"{m_label} según el reparto, por {g_label.lower()}", 470, xtitle, munit, xrange=xr)
    show(fig)

    rows = []
    for v in sorted(shown, key=str):
        sub = DS[DS[gcol] == v]
        s = bd.summarize(sub, ["ny_gpu"])
        b = s.loc[s["total_mlups"].idxmax()]
        rows.append({g_label: str(v), "Ejecuciones": len(sub), "Repartos": len(s),
                     "MLUPS GPU": rng_txt(s["gpu_mlups"]), "MLUPS CPU": rng_txt(s["cpu_mlups"]),
                     "Mejor MLUPS total": b["total_mlups"], "α del mejor": b["ny_gpu"] / float(sub["NY"].median())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Mejor MLUPS total": st.column_config.NumberColumn(format="%.1f"),
        "α del mejor": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Cada línea es la media de las ejecuciones de ese grupo en cada reparto. Si hay otra variable mezclada "
               "dentro del grupo (p. ej. agrupas por precisión con dos obstáculos), cada punto promedia esos casos.")

# ========================================================================================
# 3) BARRIDO EN DETALLE (el elegido en la barra lateral)
# ========================================================================================
S = DF[DF["sweep_id"] == sweep].copy()
SS = bd.summarize(S, ["ny_gpu"]).sort_values("ny_gpu")
r0 = S.iloc[0]
NYS = int(r0["NY"])

with tab_det:
    st.subheader("Un barrido en detalle")
    st.caption(f"{sweep}")
    st.caption(f"{r0['precision']} · malla {r0['grid']} (columnas x filas) · obstáculo {r0['obstacle']} · {r0['date']} · "
               f"{r0['steps']:.0f} pasos ({r0['warmup']:.0f} de calentamiento) · {r0['omp_threads']:.0f} hilos de CPU")
    if len(SS) < 4:
        st.warning(f"Este barrido solo tiene {len(SS)} reparto(s) ({', '.join(str(int(x)) for x in SS['ny_gpu'])} filas "
                   "GPU): es una prueba corta. Elige otro barrido en la barra lateral para ver la curva completa.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MLUPS GPU (rango)", rng_txt(SS["gpu_mlups"]), border=True)
    k2.metric("MLUPS CPU (rango)", rng_txt(SS["cpu_mlups"]), border=True)
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
                             line=dict(color=C_GPU, width=2.5), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=SS["ny_gpu"], y=SS["cpu_mlups"], mode="lines+markers", name="CPU (media)",
                             line=dict(color=C_CPU, width=2.5), marker=dict(size=8)))
    style(fig, "MLUPS de la GPU y de la CPU", 400, "filas asignadas a la GPU (la CPU hace el resto)", "MLUPS de cada lado",
          xrange=[0, NYS])
    show(fig)
    st.caption("MLUPS de un lado = celdas de ese lado / tiempo que ese lado tarda en un paso (GPU: cudaEvent, incluye la "
               "copia del halo; CPU: reloj alrededor de su cálculo). Puntos claros = cada ejecución; línea = media.")

    fig = go.Figure(go.Scatter(x=SS["ny_gpu"], y=SS["total_mlups"], mode="lines+markers", name="MLUPS total",
                               error_y=dict(type="data", array=SS["total_std"].fillna(0), visible=True, color=INK2),
                               line=dict(color=C_TOT, width=2.5), marker=dict(size=7)))
    style(fig, "MLUPS total del programa heterogéneo", 340, "filas asignadas a la GPU", "MLUPS", legend=False,
          xrange=[0, NYS])
    show(fig)

    tb = SS[["ny_gpu", "n", "gpu_mlups", "cpu_mlups", "total_mlups", "total_cv_pct", "t_gpu_ms", "t_cpu_ms", "t_step_ms"]].copy()
    tb.insert(1, "ny_cpu", NYS - tb["ny_gpu"])
    tb = tb.rename(columns={"ny_gpu": "filas GPU", "ny_cpu": "filas CPU", "n": "reps", "gpu_mlups": "MLUPS GPU",
                            "cpu_mlups": "MLUPS CPU", "total_mlups": "MLUPS total", "total_cv_pct": "CV %",
                            "t_gpu_ms": "t GPU (ms)", "t_cpu_ms": "t CPU (ms)", "t_step_ms": "t paso (ms)"})
    st.dataframe(tb, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tb.columns
                                if c not in ("filas GPU", "filas CPU", "reps")})

# ========================================================================================
# 4) CALIDAD DE LA MEDIDA (del barrido elegido)
# ========================================================================================
with tab_rep:
    st.subheader("Calidad de la medida")
    st.caption(f"{sweep}")

    cv = S.groupby("ny_gpu").agg(g_mean=("gpu_side_mlups", "mean"), g_std=("gpu_side_mlups", "std"),
                                 c_mean=("cpu_side_mlups", "mean"), c_std=("cpu_side_mlups", "std")).reset_index()
    cv["GPU"] = 100 * cv["g_std"] / cv["g_mean"]
    cv["CPU"] = 100 * cv["c_std"] / cv["c_mean"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cv["ny_gpu"].astype(str), y=cv["GPU"], name="GPU", marker=dict(color=C_GPU, cornerradius=3)))
    fig.add_trace(go.Bar(x=cv["ny_gpu"].astype(str), y=cv["CPU"], name="CPU", marker=dict(color=C_CPU, cornerradius=3)))
    style(fig, "Variabilidad entre repeticiones (CV %) en cada reparto", 360, "filas asignadas a la GPU", "CV % (desv. / media)")
    fig.update_layout(barmode="group", bargap=0.3, hovermode="closest")
    fig.update_xaxes(type="category")
    show(fig)
    st.caption("CV % = desviación típica / media de las repeticiones. Por debajo de ~2 % la media es fiable; "
               "valores altos indican que esa medida se repite mal.")

    if S["gpu_temp_max"].notna().any():
        fig = go.Figure(go.Scatter(x=S["order"], y=S["gpu_temp_max"], mode="lines+markers", name="T máx. GPU",
                                   line=dict(color=C_CPU, width=2), marker=dict(size=5)))
        style(fig, "Temperatura máxima de la GPU en cada ejecución", 320, "nº de ejecución (el orden está barajado a propósito)",
              "°C", legend=False)
        show(fig)
        st.caption("Si la temperatura sube mucho a lo largo del barrido, la GPU puede bajar su reloj y falsear las últimas medidas.")
    else:
        st.info("Este barrido no tiene datos de temperatura (nvidia-smi no estaba disponible al medir).")

# ========================================================================================
# 5) DATOS
# ========================================================================================
with tab_data:
    st.subheader("Barridos incluidos")
    inv = (DF.groupby("sweep_id").agg(Fecha=("date", "first"), Precisión=("precision", "first"), Malla=("grid", "first"),
                                      Obstáculo=("obstacle", "first"), Pasos=("steps", "first"), Ejecuciones=("order", "size"),
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
