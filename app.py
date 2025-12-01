import streamlit as st
import pandas as pd
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Government Budget Analysis (2014–2025)",
    page_icon="📊",
    layout="wide",
)

# ------------------ STYLES ------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #020617;
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .kpi-card {
        padding: 0.8rem 1rem;
        border-radius: 0.7rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.98);
        color: #e5e7eb;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .kpi-delta-pos {
        font-size: 0.8rem;
        color: #22c55e;
    }
    .kpi-delta-neg {
        font-size: 0.8rem;
        color: #ef4444;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ DATA LOADING ------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_year_column(df: pd.DataFrame) -> str | None:
    # 1) Look for a column that "looks like" year by name
    for col in df.columns:
        cl = col.lower()
        if "year" in cl or "fy" in cl or "finance year" in cl:
            return col

    # 2) Look for integer-like column with 4-digit years
    for col in df.columns:
        series = df[col]
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            vals = numeric.dropna().unique()
            if len(vals) == 0:
                continue
            if ((vals >= 2000) & (vals <= 2100)).mean() > 0.8:
                return col
        except Exception:
            continue

    return None


def split_columns(df: pd.DataFrame, year_col: str):
    num_cols = [
        c
        for c in df.columns
        if np.issubdtype(df[c].dtype, np.number)
    ]
    cat_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    # don’t offer year column as numeric measure by default
    if year_col in num_cols:
        num_cols = [c for c in num_cols if c != year_col] + [year_col]
    return num_cols, cat_cols


# ------------------ LOAD YOUR CSV ------------------
DATA_PATH = "Budget 2014-2025.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`.\n\n"
        "Make sure the CSV is in the same folder as `app.py` and named exactly like that."
    )
    st.stop()

df = df.copy()
df.columns = [c.strip() for c in df.columns]

year_col = detect_year_column(df)

if year_col is None:
    st.error(
        "Could not automatically detect a 'Year' column.\n\n"
        "Rename your year column to something like 'Year' or 'FY' and redeploy."
    )
    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.stop()

# --------- ROBUST YEAR PARSING (fixes your error) ----------
# Extract first 4-digit year starting with '20' from each cell
year_series = df[year_col].astype(str).str.extract(r"(20\\d{2})")[0]
df[year_col] = pd.to_numeric(year_series, errors="coerce")

df = df.dropna(subset=[year_col])
df[year_col] = df[year_col].astype(int)
df = df.sort_values(year_col)

if df.empty:
    st.error(
        f"After parsing, no valid years were found in column `{year_col}`.\n\n"
        "Check that it contains values like '2014-15', 'FY 2018', '2019-20 BE', etc."
    )
    st.dataframe(df.head())
    st.stop()

num_cols, cat_cols = split_columns(df, year_col)

if not num_cols:
    st.error("No numeric columns detected in the dataset to analyze.")
    st.dataframe(df.head())
    st.stop()

years = sorted(df[year_col].unique())
min_year, max_year = int(min(years)), int(max(years))

# ------------- SIDEBAR FILTERS -------------
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Financial year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

df_filtered = df[(df[year_col] >= year_range[0]) & (df[year_col] <= year_range[1])]

# choose measure (primary numeric column)
default_measure = num_cols[0]
measure = st.sidebar.selectbox(
    "Primary measure (amount column)",
    options=num_cols,
    index=num_cols.index(default_measure),
)

# convert selected measure to numeric (in case CSV stored as text with commas)
df[measure] = (
    df[measure]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False)
)
df[measure] = pd.to_numeric(df[measure], errors="coerce")
df_filtered[measure] = pd.to_numeric(df_filtered[measure], errors="coerce")

# choose breakdown dimension
dim_col = None
if cat_cols:
    dim_col_choice = st.sidebar.selectbox(
        "Break down by (category)",
        options=["(none)"] + cat_cols,
        index=0,
    )
    dim_col = None if dim_col_choice == "(none)" else dim_col_choice

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Narrow the year range to see sharper trends.")

# ------------------ HEADER ------------------
st.markdown('<div class="main-title">Government Budget Analysis</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Data: {min_year} – {max_year} • File: <code>{DATA_PATH}</code></div>',
    unsafe_allow_html=True,
)

# ------------------ KPIs ------------------
def format_amount(x):
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


current_period = df_filtered
if current_period.empty:
    latest_year = max_year
else:
    latest_year = current_period[year_col].max()

prev_year = latest_year - 1

latest_df = df[df[year_col] == latest_year]
prev_df = df[df[year_col] == prev_year] if prev_year in df[year_col].values else pd.DataFrame()

total_latest = latest_df[measure].sum()
total_prev = prev_df[measure].sum() if not prev_df.empty else np.nan

if not np.isnan(total_prev) and total_prev != 0:
    growth_pct = (total_latest - total_prev) / total_prev * 100
else:
    growth_pct = np.nan

num_entities = df_filtered[dim_col].nunique() if dim_col else len(df_filtered)

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Total (latest year)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{format_amount(total_latest)}</div>', unsafe_allow_html=True)
    st.markdown(f"<div>Year: {int(latest_year)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_kpi2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Year-on-Year change</div>', unsafe_allow_html=True)
    if np.isnan(growth_pct):
        st.markdown('<div class="kpi-value">N/A</div>', unsafe_allow_html=True)
    else:
        cls = "kpi-delta-pos" if growth_pct >= 0 else "kpi-delta-neg"
        st.markdown(
            f'<div class="kpi-value {cls}">{growth_pct:+.1f}%</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div>vs {int(prev_year) if not np.isnan(total_prev) else '-'}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_kpi3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.markdown('<div class="kpi-label">Records in selection</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-value">{len(df_filtered):,}</div>', unsafe_allow_html=True)
    if dim_col:
        st.markdown(f"<div>Unique {dim_col}: {num_entities}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ TABS ------------------
tab_overview, tab_breakdown, tab_trend, tab_table = st.tabs(
    ["📊 Overview", "🏛 Sector / Dept Breakdown", "📈 Yearly Trend", "📋 Raw Table"]
)

# ----- OVERVIEW TAB -----
with tab_overview:
    st.markdown('<div class="section-title">Overall budget trend</div>', unsafe_allow_html=True)

    agg_year = df_filtered.groupby(year_col)[measure].sum().reset_index()

    if agg_year.empty:
        st.info("No data for the selected filters.")
    else:
        st.line_chart(
            agg_year.set_index(year_col)[measure],
            height=320,
        )

    st.markdown('<div class="section-title">Top entries (by amount)</div>', unsafe_allow_html=True)

    top_n = st.slider("Show top N rows", 5, 50, 10, step=5, key="top_n_overview")

    df_sorted = df_filtered.sort_values(measure, ascending=False).head(top_n)
    st.dataframe(df_sorted, use_container_width=True)

# ----- BREAKDOWN TAB -----
with tab_breakdown:
    st.markdown(
        '<div class="section-title">Breakdown by category</div>',
        unsafe_allow_html=True,
    )

    if not dim_col:
        st.info("Select a 'Break down by (category)' column from the sidebar to see breakdowns.")
    else:
        col1, col2 = st.columns([2, 1])

        agg_dim = (
            df_filtered.groupby(dim_col)[measure]
            .sum()
            .reset_index()
            .sort_values(measure, ascending=False)
        )

        with col1:
            st.markdown(f"**Total {measure} by {dim_col}**")
            st.bar_chart(
                agg_dim.set_index(dim_col)[measure].head(15),
                height=420,
            )

        with col2:
            st.markdown("**Top categories (table)**")
            st.dataframe(
                agg_dim.head(15),
                use_container_width=True,
                height=420,
            )

        st.markdown("---")
        st.markdown(f"**Yearly trend for top {dim_col} (stacked view)**")

        top_cats = agg_dim.head(6)[dim_col].tolist()
        df_top = df_filtered[df_filtered[dim_col].isin(top_cats)]

        pivot = (
            df_top
            .groupby([year_col, dim_col])[measure]
            .sum()
            .reset_index()
            .pivot(index=year_col, columns=dim_col, values=measure)
            .fillna(0)
        )

        st.area_chart(pivot, height=360)

# ----- TREND TAB -----
with tab_trend:
    st.markdown('<div class="section-title">Detailed year-wise trend</div>', unsafe_allow_html=True)

    if not dim_col:
        st.info(
            "For more insights, select a breakdown category in the sidebar. "
            "For now, showing total trend only."
        )
        agg = df_filtered.groupby(year_col)[measure].sum().reset_index()
        st.line_chart(agg.set_index(year_col)[measure], height=360)
    else:
        selected_cat = st.selectbox(
            f"Choose a specific {dim_col} to see its trend",
            options=["(All combined)"] + sorted(df_filtered[dim_col].dropna().unique().tolist()),
        )

        if selected_cat == "(All combined)":
            agg = df_filtered.groupby(year_col)[measure].sum().reset_index()
        else:
            df_cat = df_filtered[df_filtered[dim_col] == selected_cat]
            agg = df_cat.groupby(year_col)[measure].sum().reset_index()

        st.line_chart(agg.set_index(year_col)[measure], height=360)
        st.markdown("**Yearly summary table**")
        st.dataframe(agg, use_container_width=True)

# ----- TABLE TAB -----
with tab_table:
    st.markdown('<div class="section-title">Raw filtered data</div>', unsafe_allow_html=True)
    st.caption("This is the data after applying the year filter used by all charts above.")
    st.dataframe(df_filtered, use_container_width=True, height=500)
