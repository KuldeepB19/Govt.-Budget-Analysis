import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from io import BytesIO
import re

# Page Configuration
st.set_page_config(
    page_title="Union Budget Analytics 2014-2025",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Navigation Bar */
    .nav-bar {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(30, 58, 138, 0.2);
    }
    
    .nav-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .nav-subtitle {
        color: #dbeafe;
        font-size: 1rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 5px solid #3b82f6;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.2);
        border-left-color: #f97316;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        display: inline-block;
        font-weight: 600;
    }
    
    .positive {
        background: #dcfce7;
        color: #166534;
    }
    
    .negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .neutral {
        background: #e0f2fe;
        color: #075985;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid #f97316;
        display: inline-block;
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #f97316;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.1);
    }
    
    .insight-title {
        font-weight: 700;
        color: #c2410c;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-text {
        color: #7c2d12;
        line-height: 1.8;
        font-size: 0.95rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.2);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.4);
    }
    
    /* Download Buttons */
    .download-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .download-section:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e3a8a;
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 3px dashed #3b82f6;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #eff6ff;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions for data processing
def clean_numeric_column(series):
    """Clean numeric columns that might have string values"""
    if series.dtype == 'object':
        # Remove commas and convert to float
        series = series.astype(str).str.replace(',', '').str.strip()
        series = pd.to_numeric(series, errors='coerce')
    return series

def load_and_clean_data(df):
    """Load and clean the budget dataset"""
    df = df.copy()
    
    # Clean numeric columns
    numeric_columns = [
        'Revenue (Plan)', 'Capital (Plan)', 'Total (Plan)',
        'Revenue (Non-Plan)', 'Capital (Non-Plan)', 'Total (Non-Plan)',
        'Total Plan & Non-Plan'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Extract numeric year
    if 'Year' in df.columns:
        df['Year_Numeric'] = df['Year'].astype(str).str.extract(r'(\d{4})')[0].astype(float)
    
    return df

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'df' not in st.session_state:
    st.session_state.df = None

# Load CSS
load_css()

# Navigation Bar
st.markdown("""
<div class="nav-bar">
    <h1 class="nav-title">🏛️ Union Budget Analytics Platform</h1>
    <p class="nav-subtitle">Government of India | 2014-2025 | Transparency · Insights · Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File Upload
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Select Budget CSV File",
        type=['csv'],
        help="Upload the Budget 20142025.csv file"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = load_and_clean_data(df)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df)} records")
            st.info(f"📅 Years: {int(df['Year_Numeric'].min())} - {int(df['Year_Numeric'].max())}")
            st.info(f"🏢 Ministries: {df['Ministry Name'].nunique()}")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    
    st.divider()
    
    # Quick Stats
    if st.session_state.df is not None:
        st.subheader("📊 Dataset Info")
        st.metric("Total Records", len(st.session_state.df))
        st.metric("Ministries", st.session_state.df['Ministry Name'].nunique())
        st.metric("Years Covered", int(st.session_state.df['Year_Numeric'].max() - st.session_state.df['Year_Numeric'].min() + 1))

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview Dashboard",
    "🏢 Ministry Analysis",
    "📈 Budget Forecasting",
    "🤖 AI Insights",
    "💾 Download Center"
])

# Check if data is loaded
if st.session_state.df is None:
    st.info("👆 **Please upload the Budget 20142025.csv file to get started!**")
    
    st.markdown("""
    ### 📋 About This Platform
    
    This platform analyzes the **Union Budget of India (2014-2025)** and provides:
    
    - 📊 **Interactive Dashboards** - Visualize budget trends across years
    - 🏢 **Ministry-wise Analysis** - Deep dive into each ministry's allocation
    - 📈 **Forecasting** - Predict future budget allocations using ML
    - 🤖 **AI Insights** - Ask questions in natural language
    - 💾 **Export Options** - Download filtered data in CSV/Excel
    
    ---
    
    ### 📁 Expected Dataset Structure:
    Your CSV should have these columns:
    - **Ministry Name** - Name of the ministry
    - **Year** - Budget year (e.g., 2014-15)
    - **Revenue (Plan)** - Planned revenue expenditure
    - **Capital (Plan)** - Planned capital expenditure
    - **Total (Plan)** - Total planned expenditure
    - **Revenue (Non-Plan)** - Non-plan revenue expenditure
    - **Capital (Non-Plan)** - Non-plan capital expenditure
    - **Total (Non-Plan)** - Total non-plan expenditure
    - **Total Plan & Non-Plan** - Overall total expenditure
    """)
    
    st.stop()

# Load data
df = st.session_state.df.copy()

# =====================================
# TAB 1: OVERVIEW DASHBOARD
# =====================================
with tab1:
    st.markdown('<h2 class="section-header">📊 Budget Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Year range selector
    col1, col2 = st.columns([2, 1])
    with col1:
        min_year = int(df['Year_Numeric'].min())
        max_year = int(df['Year_Numeric'].max())
        year_range = st.slider(
            "📅 Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key='overview_year_range'
        )
    
    with col2:
        metric_options = {
            'Total Plan & Non-Plan': 'Total Budget',
            'Total (Plan)': 'Plan Budget',
            'Total (Non-Plan)': 'Non-Plan Budget'
        }
        selected_metric = st.selectbox(
            "💰 Budget Category",
            options=list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
    
    # Filter data
    filtered_df = df[(df['Year_Numeric'] >= year_range[0]) & (df['Year_Numeric'] <= year_range[1])]
    
    # Calculate KPIs
    total_budget = filtered_df[selected_metric].sum()
    yearly_totals = filtered_df.groupby('Year_Numeric')[selected_metric].sum().sort_index()
    
    if len(yearly_totals) > 1:
        yoy_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2] * 100)
    else:
        yoy_growth = 0
    
    avg_annual = yearly_totals.mean()
    
    # Top ministry
    top_ministry = filtered_df.groupby('Ministry Name')[selected_metric].sum().idxmax()
    top_ministry_value = filtered_df.groupby('Ministry Name')[selected_metric].sum().max()
    
    # Display KPIs
    st.markdown("### 📈 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Total Allocation</div>
            <div class="kpi-value">₹{total_budget/1e5:.2f}L Cr</div>
            <div class="kpi-change neutral">📅 {year_range[0]}-{year_range[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        change_class = "positive" if yoy_growth >= 0 else "negative"
        arrow = "📈" if yoy_growth >= 0 else "📉"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📊 YoY Growth</div>
            <div class="kpi-value">{yoy_growth:.1f}%</div>
            <div class="kpi-change {change_class}">{arrow} {yearly_totals.index[-1]:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📉 Annual Average</div>
            <div class="kpi-value">₹{avg_annual/1e5:.2f}L Cr</div>
            <div class="kpi-change neutral">📊 Mean Allocation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        ministry_short = top_ministry[:25] + "..." if len(top_ministry) > 25 else top_ministry
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏆 Top Ministry</div>
            <div class="kpi-value" style="font-size: 1.1rem;">{ministry_short}</div>
            <div class="kpi-change positive">💰 ₹{top_ministry_value/1e5:.1f}L Cr</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Yearly Trend Chart
    st.markdown('<h3 class="section-header">📈 Budget Allocation Trends</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_totals.index,
            y=yearly_totals.values,
            mode='lines+markers',
            name=metric_options[selected_metric],
            line=dict(color='#3b82f6', width=4),
            marker=dict(size=12, color='#f97316', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig.update_layout(
            title=f"<b>{metric_options[selected_metric]} Over Time</b>",
            xaxis_title="<b>Year</b>",
            yaxis_title="<b>Amount (₹ Crores)</b>",
            hovermode='x unified',
            template='plotly_white',
            height=450,
            font=dict(size=12),
            title_font=dict(size=16)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Year-wise Data")
        yearly_display = pd.DataFrame({
            'Year': yearly_totals.index.astype(int),
            'Amount (₹ Cr)': yearly_totals.values.round(2)
        })
        st.dataframe(yearly_display, hide_index=True, use_container_width=True, height=450)
    
    # AI-Generated Insights
    st.markdown('<h3 class="section-header">💡 Automated Insights</h3>', unsafe_allow_html=True)
    
    growth_trend = "increasing" if yoy_growth > 0 else "decreasing" if yoy_growth < 0 else "stable"
    
    # Calculate additional insights
    growth_by_year = yearly_totals.pct_change() * 100
    max_growth_year = growth_by_year.idxmax()
    max_growth = growth_by_year.max()
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">🎯 Key Findings</div>
        <div class="insight-text">
            <strong>📊 Overall Trend:</strong> The {metric_options[selected_metric].lower()} from {year_range[0]} to {year_range[1]} totals <strong>₹{total_budget/1e5:.2f} Lakh Crores</strong>.<br><br>
            
            <strong>📈 Growth Pattern:</strong> Year-over-year growth is <strong>{yoy_growth:.2f}%</strong>, indicating a <strong>{growth_trend}</strong> trend. The highest single-year growth was <strong>{max_growth:.1f}%</strong> in <strong>{int(max_growth_year)}</strong>.<br><br>
            
            <strong>🏆 Top Allocation:</strong> <strong>{top_ministry}</strong> received the highest allocation with <strong>₹{top_ministry_value/1e5:.2f} Lakh Crores</strong> during this period.<br><br>
            
            <strong>💰 Annual Average:</strong> The average annual budget allocation is <strong>₹{avg_annual/1e5:.2f} Lakh Crores</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top Ministries
    st.markdown('<h3 class="section-header">🏆 Top 10 Ministries by Allocation</h3>', unsafe_allow_html=True)
    
    top_ministries = filtered_df.groupby('Ministry Name')[selected_metric].sum().sort_values(ascending=False).head(10)
    
    fig_bar = px.bar(
        y=top_ministries.index,
        x=top_ministries.values,
        orientation='h',
        title=f"<b>Top 10 Ministries - {metric_options[selected_metric]}</b>",
        labels={'x': 'Amount (₹ Crores)', 'y': 'Ministry'},
        color=top_ministries.values,
        color_continuous_scale='Blues',
        text=top_ministries.values.round(2)
    )
    
    fig_bar.update_traces(texttemplate='₹%{text:.2s}', textposition='outside')
    fig_bar.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=11),
        title_font=dict(size=16)
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Plan vs Non-Plan Breakdown
    st.markdown('<h3 class="section-header">📊 Plan vs Non-Plan Allocation</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        plan_nonplan = pd.DataFrame({
            'Category': ['Plan Budget', 'Non-Plan Budget'],
            'Amount': [
                filtered_df['Total (Plan)'].sum(),
                filtered_df['Total (Non-Plan)'].sum()
            ]
        })
        
        fig_pie = px.pie(
            plan_nonplan,
            values='Amount',
            names='Category',
            title='<b>Overall Distribution</b>',
            color_discrete_sequence=['#3b82f6', '#f97316'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, title_font=dict(size=16))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Revenue vs Capital
        rev_cap = pd.DataFrame({
            'Category': ['Revenue Expenditure', 'Capital Expenditure'],
            'Amount': [
                filtered_df['Revenue (Plan)'].sum() + filtered_df['Revenue (Non-Plan)'].sum(),
                filtered_df['Capital (Plan)'].sum() + filtered_df['Capital (Non-Plan)'].sum()
            ]
        })
        
        fig_pie2 = px.pie(
            rev_cap,
            values='Amount',
            names='Category',
            title='<b>Revenue vs Capital</b>',
            color_discrete_sequence=['#10b981', '#8b5cf6'],
            hole=0.4
        )
        fig_pie2.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie2.update_layout(height=400, title_font=dict(size=16))
        st.plotly_chart(fig_pie2, use_container_width=True)

# =====================================
# TAB 2: MINISTRY ANALYSIS
# =====================================
with tab2:
    st.markdown('<h2 class="section-header">🏢 Ministry Deep Dive Analysis</h2>', unsafe_allow_html=True)
    
    # Ministry selector
    ministries = sorted(df['Ministry Name'].unique())
    selected_ministry = st.selectbox("🔍 Select Ministry", ministries, key='ministry_selector')
    
    ministry_df = df[df['Ministry Name'] == selected_ministry]
    
    # Ministry KPIs
    st.markdown(f"### 📊 {selected_ministry}")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        total_allocation = ministry_df['Total Plan & Non-Plan'].sum()
        st.metric("💰 Total Allocation", f"₹{total_allocation/1e5:.2f}L Cr")
    
    with m2:
        years_active = ministry_df['Year_Numeric'].nunique()
        st.metric("📅 Years Covered", f"{years_active} years")
    
    with m3:
        avg_allocation = ministry_df['Total Plan & Non-Plan'].mean()
        st.metric("📊 Annual Average", f"₹{avg_allocation:.2f} Cr")
    
    with m4:
        latest_year = ministry_df['Year_Numeric'].max()
        latest_allocation = ministry_df[ministry_df['Year_Numeric'] == latest_year]['Total Plan & Non-Plan'].values[0]
        st.metric("📈 Latest Allocation", f"₹{latest_allocation:.2f} Cr", f"{int(latest_year)}")
    
    st.divider()
    
    # Ministry Trends
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📈 Multi-Year Trend Analysis")
        
        ministry_yearly = ministry_df.groupby('Year_Numeric').agg({
            'Total (Plan)': 'sum',
            'Total (Non-Plan)': 'sum',
            'Total Plan & Non-Plan': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ministry_yearly['Year_Numeric'],
            y=ministry_yearly['Total (Plan)'],
            mode='lines+markers',
            name='Plan Budget',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=ministry_yearly['Year_Numeric'],
            y=ministry_yearly['Total (Non-Plan)'],
            mode='lines+markers',
            name='Non-Plan Budget',
            line=dict(color='#f97316', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=ministry_yearly['Year_Numeric'],
            y=ministry_yearly['Total Plan & Non-Plan'],
            mode='lines+markers',
            name='Total Budget',
            line=dict(color='#10b981', width=4, dash='dash'),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"<b>{selected_ministry} - Yearly Trends</b>",
            xaxis_title="<b>Year</b>",
            yaxis_title="<b>Amount (₹ Crores)</b>",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Revenue vs Capital")
        
        rev_cap_ministry = pd.DataFrame({
            'Type': ['Revenue', 'Capital'],
            'Plan': [
                ministry_df['Revenue (Plan)'].sum(),
                ministry_df['Capital (Plan)'].sum()
            ],
            'Non-Plan': [
                ministry_df['Revenue (Non-Plan)'].sum(),
                ministry_df['Capital (Non-Plan)'].sum()
            ]
        })
        
        fig_grouped = go.Figure()
        
        fig_grouped.add_trace(go.Bar(
            name='Plan',
            x=rev_cap_ministry['Type'],
            y=rev_cap_ministry['Plan'],
            marker_color='#3b82f6',
            text=rev_cap_ministry['Plan'].round(2),
            textposition='outside'
        ))
        
        fig_grouped.add_trace(go.Bar(
            name='Non-Plan',
            x=rev_cap_ministry['Type'],
            y=rev_cap_ministry['Non-Plan'],
            marker_color='#f97316',
            text=rev_cap_ministry['Non-Plan'].round(2),
            textposition='outside'
        ))
        
        fig_grouped.update_layout(
            barmode='group',
            title='<b>Expenditure Breakdown</b>',
            yaxis_title='<b>Amount (₹ Crores)</b>',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_grouped, use_container_width=True)
    
    # Year-over-Year Growth
    st.markdown("### 📊 Year-over-Year Growth Rate")
    
    ministry_yearly_sorted = ministry_yearly.sort_values('Year_Numeric')
    ministry_yearly_sorted['YoY Growth %'] = ministry_yearly_sorted['Total Plan & Non-Plan'].pct_change() * 100
    
    fig_growth = px.bar(
        ministry_yearly_sorted[ministry_yearly_sorted['Year_Numeric'] > ministry_yearly_sorted['Year_Numeric'].min()],
        x='Year_Numeric',
        y='YoY Growth %',
        title=f'<b>{selected_ministry} - Annual Growth Rate</b>',
        color='YoY Growth %',
        color_continuous_scale='RdYlGn',
        text='YoY Growth %'
    )
    
    fig_growth.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_growth.update_layout(height=350, template='plotly_white')
    
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # Full Data Table
    st.markdown("### 📋 Complete Ministry Data")
    
    display_df = ministry_df[['Year', 'Revenue (Plan)', 'Capital (Plan)', 'Total (Plan)', 
                               'Revenue (Non-Plan)', 'Capital (Non-Plan)', 'Total (Non-Plan)', 
                               'Total Plan & Non-Plan']].sort_values('Year', ascending=False)
    
    st.dataframe(display_df, use_container_width=True, height=300)

# =====================================
# TAB 3: FORECASTING
# =====================================
with tab3:
    st.markdown('<h2 class="section-header">📈 Budget Forecasting (Next 5 Years)</h2>', unsafe_allow_html=True)
    
    st.info("🔮 Using Linear Regression to predict future budget allocations based on historical trends")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        forecast_ministry = st.selectbox(
            "🏢 Select Ministry for Forecast",
            options=['All Ministries'] + sorted(df['Ministry Name'].unique()),
            key='forecast_ministry'
        )
    
    with col2:
        forecast_metric = st.selectbox(
            "💰 Select Budget Type",
            options=['Total Plan & Non-Plan', 'Total (Plan)', 'Total (Non-Plan)'],
            key='forecast_metric'
        )
    
    # Prepare data
    if forecast_ministry == 'All Ministries':
        forecast_data = df.groupby('Year_Numeric')[forecast_metric].sum().reset_index()
    else:
        forecast_data = df[df['Ministry Name'] == forecast_ministry].groupby('Year_Numeric')[forecast_metric].sum().reset_index()
    
    forecast_data = forecast_data.sort_values('Year_Numeric')
    
    # Train model
    X = forecast_data['Year_Numeric'].values.reshape(-1, 1)
    y = forecast_data[forecast_metric].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 5 years
    last_year = int(forecast_data['Year_Numeric'].max())
    future_years = np.array([last_year + i for i in range(1, 6)]).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_data['Year_Numeric'],
            y=forecast_data[forecast_metric],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#3b82f6', width=4),
            marker=dict(size=12, color='#1e3a8a')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_years.flatten(),
            y=predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#f97316', width=4, dash='dash'),
            marker=dict(size=12, symbol='diamond', color='#ea580c')
        ))
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_years.flatten(), future_years.flatten()[::-1]]),
            y=np.concatenate([predictions * 1.1, (predictions * 0.9)[::-1]]),
            fill='toself',
            fillcolor='rgba(249, 115, 22, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='Confidence Range (±10%)'
        ))
        
        fig.update_layout(
            title=f"<b>{forecast_ministry} - 5-Year Budget Forecast</b>",
            xaxis_title="<b>Year</b>",
            yaxis_title=f"<b>{forecast_metric} (₹ Crores)</b>",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Forecast Results")
        
        forecast_df = pd.DataFrame({
            'Year': future_years.flatten().astype(int),
            'Predicted Amount (₹ Cr)': predictions.round(2),
            'Lower Bound': (predictions * 0.9).round(2),
            'Upper Bound': (predictions * 1.1).round(2)
        })
        
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)
        
        # Summary metrics
        st.markdown("### 📈 Growth Projection")
        
        total_growth = ((predictions[-1] - y[-1]) / y[-1] * 100)
        cagr = (((predictions[-1] / y[-1]) ** (1/5)) - 1) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🎯 5-Year Outlook</div>
            <div class="insight-text">
                <strong>📊 Total Growth:</strong> {total_growth:.2f}%<br>
                <strong>📈 CAGR:</strong> {cagr:.2f}%<br>
                <strong>💰 {last_year + 5} Projection:</strong> ₹{predictions[-1]:.2f} Crores<br>
                <strong>🔍 Model R² Score:</strong> {model.score(X, y):.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================
# TAB 4: AI INSIGHTS
# =====================================
with tab4:
    st.markdown('<h2 class="section-header">🤖 AI-Powered Query Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    💬 Ask questions about the budget data in natural language. The AI assistant will analyze the data and provide insights.
    
    **Example questions:**
    - Which ministry had the highest growth after 2018?
    - Show me the top 5 ministries by total allocation
    - What's the trend for Defence budget?
    - Compare Plan vs Non-Plan spending over years
    """)
    
    query = st.text_input(
        "❓ Your Question:",
        placeholder="e.g., Which ministry grew the most after 2020?",
        key='ai_query'
    )
    
    if st.button("🔍 Analyze Question", type="primary"):
        if query:
            query_lower = query.lower()
            
            # Growth analysis
            if any(word in query_lower for word in ['grow', 'growth', 'increase', 'rise']):
                st.markdown("### 📊 Growth Analysis")
                
                # Extract year if mentioned
                year_match = re.search(r'20\d{2}', query)
                threshold_year = int(year_match.group()) if year_match else 2018
                
                pre_period = df[df['Year_Numeric'] < threshold_year].groupby('Ministry Name')['Total Plan & Non-Plan'].mean()
                post_period = df[df['Year_Numeric'] >= threshold_year].groupby('Ministry Name')['Total Plan & Non-Plan'].mean()
                
                growth = ((post_period - pre_period) / pre_period * 100).sort_values(ascending=False).head(10)
                
                st.success(f"🎯 **Answer:** {growth.index[0]} showed the highest growth after {threshold_year} with **{growth.iloc[0]:.2f}%** increase.")
                
                fig = px.bar(
                    x=growth.values,
                    y=growth.index,
                    orientation='h',
                    title=f'<b>Top 10 Ministries by Growth (Post-{threshold_year})</b>',
                    labels={'x': 'Growth Rate (%)', 'y': 'Ministry'},
                    color=growth.values,
                    color_continuous_scale='RdYlGn',
                    text=growth.values
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Top/highest queries
            elif any(word in query_lower for word in ['highest', 'top', 'most', 'largest']):
                st.markdown("### 🏆 Top Performers")
                
                # Extract number
                num_match = re.search(r'\d+', query)
                top_n = int(num_match.group()) if num_match else 10
                
                totals = df.groupby('Ministry Name')['Total Plan & Non-Plan'].sum().sort_values(ascending=False).head(top_n)
                
                st.success(f"🎯 **Answer:** Top {top_n} ministries by total allocation:")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.pie(
                        values=totals.values,
                        names=totals.index,
                        title=f'<b>Top {top_n} Ministries - Budget Share</b>',
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    rank_df = pd.DataFrame({
                        'Rank': range(1, len(totals) + 1),
                        'Ministry': totals.index,
                        'Amount (₹L Cr)': (totals.values / 1e5).round(2)
                    })
                    st.dataframe(rank_df, hide_index=True, use_container_width=True, height=500)
            
            # Trend analysis
            elif 'trend' in query_lower or 'over time' in query_lower or 'years' in query_lower:
                st.markdown("### 📈 Trend Analysis")
                
                # Find ministry mentioned
                mentioned_ministry = None
                for ministry in df['Ministry Name'].unique():
                    if ministry.lower() in query_lower:
                        mentioned_ministry = ministry
                        break
                
                if mentioned_ministry:
                    ministry_trend = df[df['Ministry Name'] == mentioned_ministry].groupby('Year_Numeric')['Total Plan & Non-Plan'].sum()
                    
                    st.success(f"🎯 **Answer:** Showing budget trend for **{mentioned_ministry}**")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ministry_trend.index,
                        y=ministry_trend.values,
                        mode='lines+markers',
                        name=mentioned_ministry,
                        line=dict(color='#3b82f6', width=4),
                        marker=dict(size=12),
                        fill='tozeroy',
                        fillcolor='rgba(59, 130, 246, 0.1)'
                    ))
                    
                    fig.update_layout(
                        title=f'<b>{mentioned_ministry} - Budget Trend</b>',
                        xaxis_title='<b>Year</b>',
                        yaxis_title='<b>Amount (₹ Crores)</b>',
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional stats
                    avg_allocation = ministry_trend.mean()
                    total_allocation = ministry_trend.sum()
                    yoy_growth = ministry_trend.pct_change().mean() * 100
                    
                    st.markdown(f"""
                    <div class="insight-box">
                        <div class="insight-title">📊 Statistics</div>
                        <div class="insight-text">
                            <strong>💰 Total Allocation:</strong> ₹{total_allocation/1e5:.2f} Lakh Crores<br>
                            <strong>📊 Average Annual:</strong> ₹{avg_allocation:.2f} Crores<br>
                            <strong>📈 Avg YoY Growth:</strong> {yoy_growth:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Please mention a specific ministry name in your question.")
            
            # Compare queries
            elif 'compare' in query_lower or 'vs' in query_lower or 'versus' in query_lower:
                st.markdown("### ⚖️ Comparative Analysis")
                
                if 'plan' in query_lower and 'non-plan' in query_lower:
                    yearly_comparison = df.groupby('Year_Numeric').agg({
                        'Total (Plan)': 'sum',
                        'Total (Non-Plan)': 'sum'
                    })
                    
                    st.success("🎯 **Answer:** Plan vs Non-Plan budget comparison over years")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Plan Budget',
                        x=yearly_comparison.index,
                        y=yearly_comparison['Total (Plan)'],
                        marker_color='#3b82f6'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Non-Plan Budget',
                        x=yearly_comparison.index,
                        y=yearly_comparison['Total (Non-Plan)'],
                        marker_color='#f97316'
                    ))
                    
                    fig.update_layout(
                        barmode='group',
                        title='<b>Plan vs Non-Plan Budget Trends</b>',
                        xaxis_title='<b>Year</b>',
                        yaxis_title='<b>Amount (₹ Crores)</b>',
                        template='plotly_white',
                        height=450
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("💡 **Suggestion:** Try asking about growth, top ministries, trends, or comparisons!")
        
        else:
            st.warning("⚠️ Please enter a question.")

# =====================================
# TAB 5: DOWNLOAD CENTER
# =====================================
with tab5:
    st.markdown('<h2 class="section-header">💾 Download & Export Center</h2>', unsafe_allow_html=True)
    
    st.markdown("📥 Export your budget data in multiple formats for further analysis")
    
    # Year range for download
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        dl_year_range = st.slider(
            "Select Year Range for Export",
            min_value=int(df['Year_Numeric'].min()),
            max_value=int(df['Year_Numeric'].max()),
            value=(int(df['Year_Numeric'].min()), int(df['Year_Numeric'].max())),
            key='download_year_range'
        )
    
    with dl_col2:
        dl_ministries = st.multiselect(
            "Filter by Ministries (Optional)",
            options=sorted(df['Ministry Name'].unique()),
            default=[],
            key='download_ministries'
        )
    
    # Apply filters
    download_df = df[(df['Year_Numeric'] >= dl_year_range[0]) & (df['Year_Numeric'] <= dl_year_range[1])]
    
    if dl_ministries:
        download_df = download_df[download_df['Ministry Name'].isin(dl_ministries)]
    
    st.info(f"📊 **Records to export:** {len(download_df)} rows")
    
    st.divider()
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="download-section">
            <h3>📄 CSV Format</h3>
            <p>Filtered budget data in CSV format</p>
        </div>
        """, unsafe_allow_html=True)
        
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"budget_data_{dl_year_range[0]}_{dl_year_range[1]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("""
        <div class="download-section">
            <h3>📊 Excel Format</h3>
            <p>Data with formatting in Excel</p>
        </div>
        """, unsafe_allow_html=True)
        
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            download_df.to_excel(writer, index=False, sheet_name='Budget Data')
        
        st.download_button(
            label="⬇️ Download Excel",
            data=buffer.getvalue(),
            file_name=f"budget_data_{dl_year_range[0]}_{dl_year_range[1]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        st.markdown("""
        <div class="download-section">
            <h3>📈 Yearly Summary</h3>
            <p>Aggregated year-wise totals</p>
        </div>
        """, unsafe_allow_html=True)
        
        yearly_summary = download_df.groupby(['Year']).agg({
            'Total (Plan)': 'sum',
            'Total (Non-Plan)': 'sum',
            'Total Plan & Non-Plan': 'sum'
        }).reset_index()
        
        csv_summary = yearly_summary.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Summary",
            data=csv_summary,
            file_name=f"yearly_summary_{dl_year_range[0]}_{dl_year_range[1]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # Ministry-wise summary download
    st.markdown("### 🏢 Ministry-wise Summary Export")
    
    ministry_summary = download_df.groupby('Ministry Name').agg({
        'Total (Plan)': 'sum',
        'Total (Non-Plan)': 'sum',
        'Total Plan & Non-Plan': 'sum',
        'Revenue (Plan)': 'sum',
        'Capital (Plan)': 'sum',
        'Revenue (Non-Plan)': 'sum',
        'Capital (Non-Plan)': 'sum'
    }).reset_index().sort_values('Total Plan & Non-Plan', ascending=False)
    
    ministry_summary_csv = ministry_summary.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(ministry_summary, use_container_width=True, height=400)
    with col2:
        st.download_button(
            label="⬇️ Download Ministry Summary",
            data=ministry_summary_csv,
            file_name=f"ministry_summary_{dl_year_range[0]}_{dl_year_range[1]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # Full data preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(download_df.head(100), use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p><strong>Union Budget Analytics Platform</strong> | Government of India 2014-2025</p>
    <p>Built with ❤️ using Streamlit, Python, Pandas & Plotly</p>
    <p style='font-size: 0.85rem;'>Data Transparency · Evidence-Based Insights · Public Accountability</p>
</div>
""", unsafe_allow_html=True)
