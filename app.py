# app.py
# Streamlit app: Pronóstico de Ventas con regresiones (simples, múltiple y ponderada por R²)
# Reqs: streamlit pandas numpy scikit-learn openpyxl xlsxwriter

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Pronóstico de Ventas (Regresión)", layout="wide")

# Sugeridos (si existen en el archivo). El app NO falla si no están: usará lo que encuentre.
SUGGESTED_TARGET = "Ventas"
SUGGESTED_FEATURES = ["PIB", "Empleo", "TipoCambioPct", "Inflación"]

NUMERIC_COERCE = True


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("%", "pct")
    s = (
        s.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    return s


def _looks_transposed(df: pd.DataFrame) -> bool:
    """
    Heurística:
    - Si NO aparecen varias variables sugeridas en columnas,
      pero sí aparecen (como strings) en la primera columna o en el índice, probablemente está transpuesta.
    """
    cols_norm = set(_norm(c) for c in df.columns)
    expected_norm = set(_norm(k) for k in [SUGGESTED_TARGET] + SUGGESTED_FEATURES)

    if len(cols_norm.intersection(expected_norm)) >= 3:
        return False

    # revisar primera columna (etiquetas)
    try:
        first_col = df.columns[0]
        labels = df[first_col].astype(str).map(_norm).tolist()
        hit = len(set(labels).intersection(expected_norm))
        if hit >= 3:
            return True
    except Exception:
        pass

    # revisar índice
    try:
        idx_labels = df.index.astype(str)
        hit = len(set(_norm(x) for x in idx_labels).intersection(expected_norm))
        if hit >= 3:
            return True
    except Exception:
        pass

    return False


def _transpose_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if not _looks_transposed(df):
        return df

    df2 = df.copy()
    first_col = df2.columns[0]
    df2 = df2.set_index(first_col).T
    df2 = df2.reset_index(drop=True)
    return df2


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def _drop_na_rows(df: pd.DataFrame, needed: List[str]) -> pd.DataFrame:
    return df.dropna(subset=[c for c in needed if c in df.columns]).copy()


def _to_excel_bytes(tables: Dict[str, pd.DataFrame]) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        for name, df in tables.items():
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)[:31] or "Hoja"
            df.to_excel(writer, sheet_name=safe, index=False)
    return out.getvalue()


@dataclass
class SimpleRegResult:
    feature: str
    alpha: float
    beta: float
    r2: float
    corr: float
    yhat: Optional[float]


@dataclass
class MultiRegResult:
    r2: float
    intercept: float
    coefs: Dict[str, float]
    yhat: Optional[float]


def _simple_regressions(df: pd.DataFrame, y_col: str, x_cols: List[str], x_forecast: Dict[str, float]) -> List[SimpleRegResult]:
    out: List[SimpleRegResult] = []
    y = df[y_col].values

    for x in x_cols:
        if x not in df.columns:
            continue
        xvals = df[[x]].values
        m = LinearRegression()
        m.fit(xvals, y)
        y_pred = m.predict(xvals)
        r2 = float(r2_score(y, y_pred)) if np.isfinite(y_pred).all() else np.nan
        corr = float(np.corrcoef(df[x].values, y)[0, 1]) if df[x].notna().sum() > 1 else np.nan

        xf = x_forecast.get(x, None)
        yhat = float(m.predict(np.array([[xf]]))[0]) if xf is not None and np.isfinite(xf) else None

        out.append(SimpleRegResult(
            feature=x,
            alpha=float(m.intercept_),
            beta=float(m.coef_[0]),
            r2=r2,
            corr=corr,
            yhat=yhat
        ))
    return out


def _multiple_regression(df: pd.DataFrame, y_col: str, x_cols: List[str], x_forecast: Dict[str, float]) -> MultiRegResult:
    cols = [c for c in x_cols if c in df.columns]
    X = df[cols].values
    y = df[y_col].values
    m = LinearRegression()
    m.fit(X, y)
    y_pred = m.predict(X)
    r2 = float(r2_score(y, y_pred)) if np.isfinite(y_pred).all() else np.nan

    yhat = None
    if cols and all((c in x_forecast and np.isfinite(x_forecast[c])) for c in cols):
        Xf = np.array([[x_forecast[c] for c in cols]])
        yhat = float(m.predict(Xf)[0])

    coefs = {c: float(b) for c, b in zip(cols, m.coef_)}
    return MultiRegResult(
        r2=r2,
        intercept=float(m.intercept_),
        coefs=coefs,
        yhat=yhat
    )


def _weighted_forecast_by_r2(simples: List[SimpleRegResult]) -> Tuple[Optional[float], pd.DataFrame]:
    """
    Pronóstico ponderado por R² usando pronósticos de regresiones simples:
    peso_i = max(R²_i, 0) / suma(max(R², 0))
    """
    rows = []
    for r in simples:
        w_raw = max(r.r2, 0.0) if (r.r2 is not None and np.isfinite(r.r2)) else 0.0
        rows.append((r.feature, r.r2, r.yhat, w_raw))

    dfw = pd.DataFrame(rows, columns=["Variable", "R2", "Pronostico_Simple", "Peso_sin_norm"])
    denom = dfw["Peso_sin_norm"].sum()
    if denom <= 0 or dfw["Pronostico_Simple"].isna().all():
        dfw["Peso"] = 0.0
        dfw["Aporte"] = np.nan
        return None, dfw

    dfw["Peso"] = dfw["Peso_sin_norm"] / denom
    dfw["Aporte"] = dfw["Peso"] * dfw["Pronostico_Simple"]
    yhat_w = float(dfw["Aporte"].sum(skipna=True))
    return yhat_w, dfw


def _read_any(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def _pick_defaults(cols: List[str]) -> Tuple[Optional[str], List[str]]:
    """Sugiere target y X con base en nombres sugeridos, pero SIEMPRE dentro de las columnas disponibles."""
    target = SUGGESTED_TARGET if SUGGESTED_TARGET in cols else (cols[0] if cols else None)

    # X sugeridas que existan y no sean target
    xs = [c for c in SUGGESTED_FEATURES if c in cols and c != target]
    # Si no hay sugeridas, usar el resto
    if not xs:
        xs = [c for c in cols if c != target]
    return target, xs


# -----------------------------
# UI
# -----------------------------
st.title("Pronóstico de Ventas con Regresiones (simples, múltiple y ponderada por R²)")

with st.sidebar:
    st.header("1) Cargar históricos")
    hist_file = st.file_uploader("Archivo histórico (CSV o Excel)", type=["csv", "xlsx", "xls"])

    st.divider()
    st.header("2) Pronósticos (X)")
    mode = st.radio("¿Cómo vas a ingresar pronósticos?", ["Manual (1 periodo)", "Cargar archivo de pronósticos"], index=0)

    forecast_file = None
    if mode == "Cargar archivo de pronósticos":
        forecast_file = st.file_uploader("Archivo pronósticos (CSV o Excel)", type=["csv", "xlsx", "xls"])

# -----------------------------
# Load data
# -----------------------------
if hist_file is None:
    st.info("Carga tu archivo histórico para empezar.")
    st.stop()

raw = _read_any(hist_file)
raw = _transpose_if_needed(raw)
df = raw.copy()

# Coerción numérica (solo para columnas que vamos a usar; primero detectamos columnas disponibles)
available_cols = list(df.columns)

# Selector de variables (dependiente e independientes) basado en lo que subió el usuario
with st.sidebar:
    st.divider()
    st.header("3) Selección de variables")
    st.caption("Elige la variable dependiente (Y) y las independientes (X) ENTRE las columnas encontradas en tu archivo.")
    default_y, default_xs = _pick_defaults(available_cols)

    y_col = st.selectbox("Variable dependiente (Y)", options=available_cols, index=available_cols.index(default_y) if default_y in available_cols else 0)
    x_cols = st.multiselect("Variables independientes (X)", options=[c for c in available_cols if c != y_col], default=[c for c in default_xs if c != y_col])

if not x_cols:
    st.warning("Selecciona al menos 1 variable independiente (X) para continuar.")
    st.stop()

# Coerce numeric for chosen cols
needed_cols = [y_col] + x_cols
if NUMERIC_COERCE:
    df = _coerce_numeric(df, needed_cols)

# Aviso: si faltan datos en columnas seleccionadas, se depuran filas en modelos (no se detiene)
st.subheader("Históricos (limpios)")
st.dataframe(df, use_container_width=True)

# -----------------------------
# Build forecast inputs
# -----------------------------
x_forecasts: Dict[str, float] = {}
forecast_df: Optional[pd.DataFrame] = None

if mode == "Manual (1 periodo)":
    with st.sidebar:
        st.divider()
        st.subheader("Pronósticos manuales (1 periodo)")
        for x in x_cols:
            default_val = float(df[x].dropna().iloc[-1]) if x in df.columns and df[x].dropna().shape[0] else 0.0
            x_forecasts[x] = float(st.number_input(f"{x}", value=default_val, step=0.1, format="%.6f"))
else:
    if forecast_file is None:
        st.info("Carga el archivo de pronósticos para continuar.")
        st.stop()

    fraw = _read_any(forecast_file)
    fraw = _transpose_if_needed(fraw)
    forecast_df = fraw.copy()

    # Coerce numeric only for X seleccionadas que existan en pronósticos
    if NUMERIC_COERCE:
        forecast_df = _coerce_numeric(forecast_df, [c for c in x_cols if c in forecast_df.columns])

    st.subheader("Pronósticos cargados (X)")
    st.dataframe(forecast_df, use_container_width=True)

    # Tomar la primera fila como escenario principal (solo con X disponibles)
    first = forecast_df.iloc[0].to_dict()
    for x in x_cols:
        if x in first and pd.notna(first[x]):
            x_forecasts[x] = float(first[x])

# -----------------------------
# Compute stats (solo con lo que existe / fue seleccionado)
# -----------------------------
model_df = df.dropna(subset=[y_col] + [c for c in x_cols if c in df.columns]).copy()

if model_df.shape[0] < 3:
    st.error("No hay suficientes observaciones (filas) con datos completos para estimar regresiones. Revisa NA / celdas vacías.")
    st.stop()

# Correlaciones con Y (solo para X seleccionadas)
corr_rows = []
for x in x_cols:
    if x in model_df.columns and y_col in model_df.columns:
        corr = model_df[[x, y_col]].corr().iloc[0, 1]
        corr_rows.append({"Variable": x, f"Correlacion_con_{y_col}": float(corr)})
corr_table = pd.DataFrame(corr_rows)

# Regresiones simples
simples = _simple_regressions(model_df, y_col, x_cols, x_forecasts)

simples_table = pd.DataFrame([{
    "Variable": r.feature,
    "Alpha_(interseccion)": r.alpha,
    "Beta_(pendiente)": r.beta,
    "R2": r.r2,
    "Correlacion": r.corr,
    "Pronostico_Simple": r.yhat
} for r in simples])

# Regresión múltiple
multi = _multiple_regression(model_df, y_col, x_cols, x_forecasts)
multi_table = pd.DataFrame([{
    "R2": multi.r2,
    "Interseccion": multi.intercept,
    **{f"Coef_{k}": v for k, v in multi.coefs.items()},
    "Pronostico_Multiple": multi.yhat
}])

# Pronóstico ponderado por R²
weighted_yhat, weights_table = _weighted_forecast_by_r2(simples)
summary = pd.DataFrame([{
    "Y": y_col,
    "Xs": ", ".join(x_cols),
    "Pronostico_Ponderado_R2": weighted_yhat,
    "Pronostico_Regresion_Multiple": multi.yhat,
    "R2_Multiple": multi.r2
}])

# -----------------------------
# Display
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader(f"Correlaciones con {y_col}")
    st.dataframe(corr_table, use_container_width=True)

with c2:
    st.subheader("Regresiones simples (β, α, R²) + pronóstico")
    st.dataframe(simples_table, use_container_width=True)

st.subheader("Ponderación por R² (pronóstico combinado desde regresiones simples)")
st.dataframe(weights_table, use_container_width=True)

st.subheader("Regresión múltiple")
st.dataframe(multi_table, use_container_width=True)

st.subheader("Resumen")
st.dataframe(summary, use_container_width=True)

# -----------------------------
# Batch mode: si cargaron archivo de pronósticos, calcular pronósticos por fila
# -----------------------------
batch_table = None
if forecast_df is not None:
    # Solo X seleccionadas que existan en el archivo de pronósticos y en históricos
    cols_present = [c for c in x_cols if c in forecast_df.columns and c in model_df.columns]
    if cols_present:
        batch_rows = []

        y = model_df[y_col].values
        multi_model = LinearRegression().fit(model_df[cols_present].values, y)

        simple_models = {}
        simple_r2 = {}
        for x in cols_present:
            m = LinearRegression().fit(model_df[[x]].values, y)
            ypred = m.predict(model_df[[x]].values)
            simple_models[x] = m
            simple_r2[x] = max(float(r2_score(y, ypred)), 0.0)

        denom = sum(simple_r2.values()) if sum(simple_r2.values()) > 0 else 0.0
        weights = {x: (simple_r2[x] / denom if denom > 0 else 0.0) for x in cols_present}

        for i, row in forecast_df.iterrows():
            # Requiere todas las X presentes para ese escenario
            if any(pd.isna(row.get(x, np.nan)) for x in cols_present):
                continue

            yhat_sim = {x: float(simple_models[x].predict(np.array([[float(row[x])]]))[0]) for x in cols_present}
            yhat_w = float(sum(weights[x] * yhat_sim[x] for x in cols_present)) if denom > 0 else np.nan

            Xf = np.array([[float(row[x]) for x in cols_present]])
            yhat_m = float(multi_model.predict(Xf)[0])

            batch_rows.append({
                "Fila": int(i),
                **{x: float(row[x]) for x in cols_present},
                "Pronostico_Ponderado_R2": yhat_w,
                "Pronostico_Multiple": yhat_m
            })

        batch_table = pd.DataFrame(batch_rows)
        st.subheader("Pronósticos por escenario (archivo de pronósticos)")
        st.dataframe(batch_table, use_container_width=True)

# -----------------------------
# Download
# -----------------------------
tables = {
    "Correlaciones": corr_table,
    "Regresiones_Simples": simples_table,
    "Pesos_R2": weights_table,
    "Regresion_Multiple": multi_table,
    "Resumen": summary,
}
if batch_table is not None:
    tables["Batch_Pronosticos"] = batch_table

xlsx = _to_excel_bytes(tables)
st.download_button(
    label="Descargar resultados en Excel",
    data=xlsx,
    file_name="resultados_pronostico_regresion.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
