import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Government Budget Analytics",
    page_icon="🇮🇳",
    layout="wide",
)

# ================== THEME & GLOBAL STYLES ==================
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# Theme toggle
with st.sidebar:
    st.caption("Theme")
    dark_mode = st.toggle("Dark mode", value=(st.session_state.theme == "Dark"))
    st.session_state.theme = "Dark" if dark_mode else "Light"

if st.session_state.theme == "Dark":
    BG = "#020617"
    CARD_BG = "#020617"
    TEXT = "#e5e7eb"
    SUBTEXT = "#9ca3af"
    BORDER = "#1f2937"
else:
    BG = "#f9fafb"
    CARD_BG = "#ffffff"
    TEXT = "#020617"
    SUBTEXT = "#4b5563"
    BORDER = "#e5e7eb"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .main-header {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.3rem;
    }}
    .main-header-title {{
        font-size: 1.8rem;
        font-weight: 700;
    }}
    .main-header-badge {{
        font-size: 0.85rem;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        border: 1px solid {BORDER};
        background: {CARD_BG};
        color: {SUBTEXT};
    }}
    .subtitle {{
        color: {SUBTEXT};
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(3,minmax(0,1fr));
        gap: 0.9rem;
        margin: 0.7rem 0 1rem 0;
    }}
    .kpi-card {{
        padding: 0.8rem 1rem;
        border-radius: 0.8rem;
        border: 1px solid {BORDER};
        background: {CARD_BG};
        transition: transform 0.08s ease-out, box-shadow 0.08s ease-out;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 18px rgba(15,23,42,0.18);
    }}
    .kpi-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {SUBTEXT};
        margin-bottom: 0.2rem;
    }}
    .kpi-value {{
        font-size: 1.25rem;
        font-weight: 600;
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: {SUBTEXT};
        margin-top: 0.15rem;
    }}
    .section-title {{
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0.6rem 0 0.3rem 0;
    }}
    .ai-bubble {{
        padding: 0.7rem 0.8rem;
        border-radius: 0.8rem;
        border: 1px solid {BORDER};
        background: {CARD_BG};
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    .pill {{
        display:inline-block;
        padding:0.2rem 0.5rem;
        border-radius:999px;
        border:1px solid {BORDER};
        font-size:0.72rem;
        color:{SUBTEXT};
        margin-right:0.25rem;
        margin-bottom:0.25rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== DATA LOADING & DETECTION ==================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_year_column(df: pd.DataFrame) -> str | None:
    # 1) By name
    for col in df.columns:
        cl = col.lower()
        if "year" in cl or "fy" in cl or "financial year" in cl:
            return col
    # 2) by numeric 4-digit pattern
    for col in df.columns:
        try:
            s = df[col].astype(str).str.extract(r"(20\d{2})")[0]
            vals = pd.to_numeric(s, errors="coerce").dropna().unique()
            if len(vals) == 0:
                continue
            if ((vals >= 2000) & (vals <= 2100)).mean() > 0.8:
                return col
        except Exception:
            continue
    return None


def detect_column(df: pd.DataFrame, keywords, exclude=None):
    exclude = exclude or []
    for col in df.columns:
        cl = col.lower()
        if any(kw in cl for kw in keywords) and all(ex not in cl for ex in exclude):
            return col
    return None


DATA_PATH = "Budget 2014-2025.csv"
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"CSV `{DATA_PATH}` not found. Place it next to app.py in your repo."
    )
    st.stop()

df = df.copy()
df.columns = [c.strip() for c in df.columns]

year_col = detect_year_column(df)
if not year_col:
    st.error("Could not detect a Year/FY column. Please rename your year column to include 'Year' or 'FY'.")
    st.dataframe(df.head())
    st.stop()

# Parse year robustly
raw_year = df[year_col].astype(str)
year_extracted = raw_year.str.extract(r"(20\d{2})")[0]
df[year_col] = pd.to_numeric(year_extracted, errors="coerce")
df = df.dropna(subset=[year_col])
df[year_col] = df[year_col].astype(int)
df = df.sort_values(year_col)

if df.empty:
    st.error("No valid years found after parsing. Check the Year column format.")
    st.dataframe(raw_year.head(20).to_frame(name=year_col))
    st.stop()

# Numeric & category cols
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
cat_cols = [c for c in df.columns if c not in num_cols]

# Detect useful dims
ministry_col = detect_column(df, ["ministry", "department", "head"])
state_col = detect_column(df, ["state", "ut", "union territory"])
be_col = detect_column(df, ["be", "budget estimate"])
re_col = detect_column(df, ["re", "revised estimate"])
ae_col = detect_column(df, ["ae", "actual"], exclude=["re"])
total_default = None

# Choose a default numeric column as main measure (prefer BE/Total/Expenditure)
candidates = [
    be_col,
    detect_column(df, ["total", "expenditure", "outlay"]),
]
for c in candidates:
    if c and c in num_cols:
        total_default = c
        break
if total_default is None:
    total_default = [c for c in num_cols if c != year_col][0]

# Clean numeric columns: remove commas, spaces
for c in num_cols:
    if df[c].dtype == object or df[c].dtype == "O":
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

# recompute num_cols after coercion
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

years = sorted(df[year_col].unique())
min_year, max_year = int(min(years)), int(max(years))

# ================== SIDEBAR FILTERS ==================
st.sidebar.header("Global Filters")

year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

measure = st.sidebar.selectbox(
    "Primary amount measure",
    options=[c for c in num_cols if c != year_col],
    index=[c for c in num_cols if c != year_col].index(total_default),
)

dim_col = st.sidebar.selectbox(
    "Break down by",
    options=["(none)"] + cat_cols,
    index=0,
)
dim_col = None if dim_col == "(none)" else dim_col

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Budget 2014–2025 (custom CSV)")

# Filtered dataframe
df_filtered = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])].copy()

# ================== HEADER ==================
st.markdown(
    f"""
    <div class="main-header">
        <div class="main-header-title">Union Budget Analytics</div>
        <div class="main-header-badge">Government of India · Experimental</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="subtitle">Analysis for {year_range[0]}–{year_range[1]} · Measure: <code>{measure}</code></div>',
    unsafe_allow_html=True,
)

# fake navbar using radio
nav_pages = [
    "Overview",
    "Ministries",
    "States" if state_col else None,
    "Forecasting",
    "AI Insights",
    "Downloads",
    "About",
]
nav_pages = [p for p in nav_pages if p]

page = st.radio(
    "Navigation",
    nav_pages,
    horizontal=True,
    label_visibility="collapsed",
)

# ================== SMALL HELPERS ==================
def fmt_amt(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "-"
    if abs(x) >= 1e7:
        return f"{x/1e7:,.1f} Cr"
    elif abs(x) >= 1e5:
        return f"{x/1e5:,.1f} L"
    else:
        return f"{x:,.0f}"


def yoy_growth(s: pd.Series):
    if len(s) < 2 or s.iloc[-2] == 0 or pd.isna(s.iloc[-2]):
        return np.nan
    return (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100


# ================== KPIs (shown on most pages) ==================
latest_year = df_filtered[year_col].max()
prev_year = latest_year - 1

latest_df = df_filtered[df_filtered[year_col] == latest_year]
prev_df = df_filtered[df_filtered[year_col] == prev_year]

total_latest = latest_df[measure].sum()
total_prev = prev_df[measure].sum() if not prev_df.empty else np.nan
growth_pct = (
    (total_latest - total_prev) / total_prev * 100
    if total_prev not in [0, np.nan] and not np.isnan(total_prev)
    else np.nan
)

if dim_col:
    top_cat = (
        latest_df.groupby(dim_col)[measure].sum().sort_values(ascending=False).head(1)
    )
    top_cat_name = top_cat.index[0]
    top_cat_val = top_cat.iloc[0]
else:
    top_cat_name = None
    top_cat_val = None

kpi_html = "<div class='kpi-row'>"

# KPI 1
kpi_html += f"""
<div class='kpi-card'>
  <div class='kpi-label'>Total in {latest_year}</div>
  <div class='kpi-value'>{fmt_amt(total_latest)}</div>
  <div class='kpi-sub'>Primary measure: {measure}</div>
</div>
"""

# KPI 2
if not np.isnan(growth_pct):
    cls = "kpi-delta-pos" if growth_pct >= 0 else "kpi-delta-neg"
    sign = "+" if growth_pct >= 0 else ""
    kpi_html += f"""
    <div class='kpi-card'>
      <div class='kpi-label'>YoY Change vs {prev_year}</div>
      <div class='kpi-value {cls}'>{sign}{growth_pct:.1f}%</div>
      <div class='kpi-sub'>Previous: {fmt_amt(total_prev)}</div>
    </div>
    """
else:
    kpi_html += f"""
    <div class='kpi-card'>
      <div class='kpi-label'>YoY Change</div>
      <div class='kpi-value'>N/A</div>
      <div class='kpi-sub'>Not enough data</div>
    </div>
    """

# KPI 3
if top_cat_name and dim_col:
    kpi_html += f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Top {dim_col} in {latest_year}</div>
      <div class='kpi-value'>{top_cat_name}</div>
      <div class='kpi-sub'>Total: {fmt_amt(top_cat_val)}</div>
    </div>
    """
else:
    kpi_html += f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Records in selection</div>
      <div class='kpi-value'>{len(df_filtered):,}</div>
      <div class='kpi-sub'>Years: {year_range[0]}–{year_range[1]}</div>
    </div>
    """

kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ================== PAGE: OVERVIEW ==================
if page == "Overview":
    st.markdown("<div class='section-title'>Overall trend</div>", unsafe_allow_html=True)
    agg_year = df_filtered.groupby(year_col)[measure].sum().reset_index()
    if agg_year.empty:
        st.info("No data for the selected filters.")
    else:
        st.line_chart(agg_year.set_index(year_col)[measure], height=320)

    st.markdown("<div class='section-title'>Top entries (by amount)</div>", unsafe_allow_html=True)
    top_n = st.slider("Show top N rows", 5, 50, 10, step=5, key="top_n_overview")
    df_sorted = df_filtered.sort_values(measure, ascending=False).head(top_n)
    st.dataframe(df_sorted, use_container_width=True)

    # Small auto-insight text
    st.markdown("<div class='section-title'>Key insights</div>", unsafe_allow_html=True)
    bullets = []
    if not agg_year.empty and len(agg_year) >= 2:
        first_val = agg_year[measure].iloc[0]
        last_val = agg_year[measure].iloc[-1]
        if first_val != 0:
            change_total = (last_val - first_val) / first_val * 100
            direction = "increased" if change_total >= 0 else "decreased"
            bullets.append(
                f"Total {measure} has **{direction} by {change_total:+.1f}%** between {agg_year[year_col].iloc[0]} and {agg_year[year_col].iloc[-1]}."
            )
    if dim_col and not df_filtered.empty:
        by_dim_latest = latest_df.groupby(dim_col)[measure].sum().sort_values(ascending=False)
        if len(by_dim_latest) >= 2:
            top1, top2 = by_dim_latest.index[0], by_dim_latest.index[1]
            bullets.append(
                f"In {latest_year}, **{top1}** has the highest allocation, followed by **{top2}**."
            )

    if not bullets:
        st.caption("No strong patterns detected yet – adjust filters or pick a different measure.")
    else:
        for b in bullets:
            st.markdown(f"• {b}")

# ================== PAGE: MINISTRIES ==================
elif page == "Ministries":
    st.markdown("<div class='section-title'>Ministry / Department view</div>", unsafe_allow_html=True)

    if not ministry_col:
        st.info("No Ministry/Department column detected. Make sure one column name contains 'Ministry' or 'Department'.")
    else:
        ministries = sorted(df_filtered[ministry_col].dropna().unique().tolist())
        if not ministries:
            st.info("No ministry records in current filter.")
        else:
            selected_min = st.selectbox("Select ministry / department", ministries)
            df_min = df_filtered[df_filtered[ministry_col] == selected_min]

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Yearly trend for {selected_min}**")
                agg_min = df_min.groupby(year_col)[measure].sum().reset_index()
                st.line_chart(agg_min.set_index(year_col)[measure], height=320)

            with col2:
                st.markdown("**Snapshot for latest year**")
                latest_min = df_min[df_min[year_col] == latest_year]
                st.dataframe(latest_min.sort_values(measure, ascending=False).head(15), use_container_width=True)

            if be_col and re_col and ae_col:
                st.markdown("<div class='section-title'>BE / RE / AE comparison (latest year)</div>", unsafe_allow_html=True)
                cols = [be_col, re_col, ae_col]
                latest_min_be = latest_min[[ministry_col, be_col, re_col, ae_col, year_col]]
                df_be = latest_min_be.groupby(ministry_col)[cols].sum()
                st.bar_chart(df_be.T, height=320)

            st.markdown("**All records for this ministry (filtered years)**")
            st.dataframe(df_min, use_container_width=True)

# ================== PAGE: STATES ==================
elif page == "States":
    st.markdown("<div class='section-title'>State-wise analysis</div>", unsafe_allow_html=True)
    if not state_col:
        st.info("No State column detected.")
    else:
        agg_state = (
            df_filtered.groupby(state_col)[measure]
            .sum()
            .reset_index()
            .sort_values(measure, ascending=False)
        )
        st.bar_chart(agg_state.set_index(state_col)[measure].head(25), height=360)
        st.markdown("**State-wise table**")
        st.dataframe(agg_state, use_container_width=True)
        st.caption("You can later plug this into a choropleth map using India State GeoJSON.")

# ================== PAGE: FORECASTING ==================
elif page == "Forecasting":
    st.markdown("<div class='section-title'>Budget forecasting</div>", unsafe_allow_html=True)
    st.caption("Lightweight ML model (Linear Regression) — Streamlit Cloud friendly.")

    agg = df.groupby(year_col)[measure].sum().reset_index().sort_values(year_col)

    if len(agg) < 3:
        st.warning("Not enough data for forecasting (need at least 3 years).")
    else:
        from sklearn.linear_model import LinearRegression

        X = agg[[year_col]].values
        y = agg[measure].values
        model = LinearRegression().fit(X, y)

        last_year_all = agg[year_col].max()
        future_years = np.array(range(last_year_all + 1, last_year_all + 6)).reshape(-1, 1)
        preds = model.predict(future_years)

        df_forecast = agg.copy()
        df_forecast["Forecast"] = np.nan
        forecast_part = pd.DataFrame({year_col: future_years.flatten(), "Forecast": preds})
        df_forecast = pd.concat([df_forecast, forecast_part], ignore_index=True).sort_values(year_col)
        df_forecast = df_forecast.set_index(year_col)

        st.line_chart(df_forecast, height=360)

        st.markdown("**Forecast values (next 5 years)**")
        table = pd.DataFrame(
            {"Year": future_years.flatten(), f"Forecast_{measure}": preds}
        )
        st.dataframe(table, use_container_width=True)

        last_hist = y[-1]
        growth = (preds[-1] - last_hist) / last_hist * 100 if last_hist != 0 else np.nan
        if not np.isnan(growth):
            st.markdown(
                f"> By **{future_years.flatten()[-1]}**, {measure} is projected to be **{fmt_amt(preds[-1])}**, "
                f"about **{growth:+.1f}% higher** than {agg[year_col].iloc[-1]}."
            )

# ================== PAGE: AI INSIGHTS ==================
elif page == "AI Insights":
    st.markdown("<div class='section-title'>AI-style Insights Assistant (rule-based)</div>", unsafe_allow_html=True)
    st.caption("Lightweight rule-based assistant (no external API). Ask something about trends.")

    q = st.text_input("Ask a question (e.g., 'Which ministry grew the most after 2020?')", "")

    if q.strip():
        q_low = q.lower()
        response_lines = []

        # Example: biggest increase after a year
        if "which" in q_low and ("highest" in q_low or "largest" in q_low or "most" in q_low) and ("increase" in q_low or "growth" in q_low):
            import re
            yrs_in_q = re.findall(r"20\d{2}", q_low)
            base_year = int(yrs_in_q[0]) if yrs_in_q else min_year
            if not dim_col and ministry_col:
                target_dim = ministry_col
            else:
                target_dim = dim_col or ministry_col

            if not target_dim:
                response_lines.append("I couldn't find a category column (like Ministry or State) to compare growth.")
            else:
                df_window = df[df[year_col] >= base_year].copy()
                if df_window.empty:
                    response_lines.append(f"No data found from {base_year} onwards.")
                else:
                    start_y = df_window[year_col].min()
                    end_y = df_window[year_col].max()
                    start_agg = df_window[df_window[year_col] == start_y].groupby(target_dim)[measure].sum()
                    end_agg = df_window[df_window[year_col] == end_y].groupby(target_dim)[measure].sum()
                    joined = pd.concat([start_agg, end_agg], axis=1, keys=["start", "end"]).dropna()
                    joined = joined[joined["start"] != 0]
                    joined["growth_pct"] = (joined["end"] - joined["start"]) / joined["start"] * 100
                    if joined.empty:
                        response_lines.append("Not enough data to compute growth for that period.")
                    else:
                        best = joined.sort_values("growth_pct", ascending=False).iloc[0]
                        name = joined.sort_values("growth_pct", ascending=False).index[0]
                        response_lines.append(
                            f"From **{start_y}** to **{end_y}**, **{name}** shows the **highest growth** in {measure}, "
                            f"about **{best['growth_pct']:+.1f}%**."
                        )

        # Example: total highest in latest year
        if any(w in q_low for w in ["total", "allocation", "spending", "expenditure"]) and ("for" in q_low or "of" in q_low):
            target_dim = ministry_col or state_col or dim_col
            if target_dim:
                latest_group = latest_df.groupby(target_dim)[measure].sum().sort_values(ascending=False)
                if not latest_group.empty:
                    response_lines.append(
                        f"In **{latest_year}**, the highest {measure} is for **{latest_group.index[0]}** "
                        f"with about **{fmt_amt(latest_group.iloc[0])}**."
                    )

        if not response_lines:
            agg_year = df_filtered.groupby(year_col)[measure].sum().reset_index()
            if not agg_year.empty:
                first_val = agg_year[measure].iloc[0]
                last_val = agg_year[measure].iloc[-1]
                if first_val != 0:
                    change_total = (last_val - first_val) / first_val * 100
                    direction = "higher" if change_total >= 0 else "lower"
                    response_lines.append(
                        f"Across {agg_year[year_col].iloc[0]}–{agg_year[year_col].iloc[-1]}, total {measure} is about "
                        f"**{change_total:+.1f}% {direction}**."
                    )
            else:
                response_lines.append("I couldn't detect a clear trend with the current filters.")

        st.markdown("<div class='ai-bubble'>", unsafe_allow_html=True)
        for line in response_lines:
            st.markdown(line)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Example questions:**")
    st.markdown(
        "<span class='pill'>Which ministry saw the highest growth after 2020?</span>"
        "<span class='pill'>Which sector has the highest allocation in the latest year?</span>"
        "<span class='pill'>How has total expenditure changed over time?</span>",
        unsafe_allow_html=True,
    )

# ================== PAGE: DOWNLOADS ==================
elif page == "Downloads":
    st.markdown("<div class='section-title'>Download data</div>", unsafe_allow_html=True)
    st.caption("Export the filtered dataset or aggregated views.")

    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_data,
        file_name=f"budget_filtered_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv",
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_filtered.to_excel(writer, sheet_name="FilteredData", index=False)
    st.download_button(
        "Download filtered data as Excel",
        data=buffer.getvalue(),
        file_name=f"budget_filtered_{year_range[0]}_{year_range[1]}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("<div class='section-title'>Quick aggregated view</div>", unsafe_allow_html=True)
    agg_year = df_filtered.groupby(year_col)[measure].sum().reset_index()
    st.dataframe(agg_year, use_container_width=True)

# ================== PAGE: ABOUT ==================
elif page == "About":
    st.markdown("<div class='section-title'>About this platform</div>", unsafe_allow_html=True)
    st.markdown(
        """
        This experimental dashboard is designed as a **multi-audience platform** for analysing the Union Budget:

        - 🟢 **Clean economic dashboard** – macro trends, aggregates, KPIs  
        - 🔵 **Interactive research tool for students** – drill-down, filters, downloads  
        - 🟡 **Public transparency view** – simple visuals & narrative insights  
        - 🔴 **Policy analyst's lens** – BE/RE/AE comparisons (if columns exist), ministry focus, forecasting  

        **Technical stack**

        - Streamlit for the web UI  
        - `pandas` & `numpy` for data wrangling  
        - `scikit-learn` (LinearRegression) for simple forecasting  

        You can extend this by:
        - Plugging in a proper LLM (OpenAI, etc.) for a real AI assistant  
        - Adding a choropleth map using India State GeoJSON  
        - Styling it with official GoI colours and insignia
        """
    )

    st.markdown("<div class='section-title'>Detected dataset structure</div>", unsafe_allow_html=True)
    st.write("**Columns:**", list(df.columns))
    st.write("**Year column:**", year_col)
    st.write("**Measure (current):**", measure)
    st.write("**Ministry column:**", ministry_col)
    st.write("**State column:**", state_col)
    st.write("**BE/RE/AE (if detected):**", {"BE": be_col, "RE": re_col, "AE": ae_col})

    st.markdown("<div class='section-title'>Data preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)
