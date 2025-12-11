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
    /* Theme Variables */
    :root {
        --primary-color: #FF6B35;
        --secondary-color: #004E89;
        --background-light: #F7F9FC;
        --background-dark: #1E1E1E;
        --text-light: #2C3E50;
        --text-dark: #ECEFF4;
    }
    
    /* Main Container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Navigation Bar */
    .nav-bar {
        background: linear-gradient(135deg, #004E89 0%, #1A659E 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .nav-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .nav-subtitle {
        color: #E8F1F5;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0.3rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #004E89;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #004E89;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-change {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
    }
    
    .positive {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .negative {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #004E89;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FF6B35;
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(135deg, #FFF5F0 0%, #FFE8DC 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    
    .insight-title {
        font-weight: 700;
        color: #C2410C;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        color: #78350F;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #004E89 0%, #1A659E 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0,78,137,0.3);
    }
    
    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Dark Mode Adjustments */
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: #2D3748;
            color: white;
        }
        .kpi-value {
            color: #63B3ED;
        }
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 2px dashed #004E89;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Dataset Detection Functions
def detect_year_column(df):
    """Detect year column from various formats"""
    patterns = [
        r'year', r'fy', r'fiscal', r'budget.*year', r'yr',
        r'\d{4}-\d{2}', r'\d{4}'
    ]
    for col in df.columns:
        col_lower = str(col).lower()
        for pattern in patterns:
            if re.search(pattern, col_lower):
                return col
    return None

def detect_category_columns(df):
    """Detect ministry/department columns"""
    candidates = {
        'ministry': None,
        'department': None,
        'scheme': None,
        'state': None
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        if 'ministry' in col_lower and not candidates['ministry']:
            candidates['ministry'] = col
        elif 'department' in col_lower and not candidates['department']:
            candidates['department'] = col
        elif 'scheme' in col_lower and not candidates['scheme']:
            candidates['scheme'] = col
        elif 'state' in col_lower and not candidates['state']:
            candidates['state'] = col
    
    return candidates

def detect_numeric_columns(df, exclude_cols):
    """Detect expenditure columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_cols]

def parse_year_column(df, year_col):
    """Parse year column to extract numeric year"""
    df_copy = df.copy()
    
    if df_copy[year_col].dtype == 'object':
        # Handle formats like "2016-17", "FY 2019", etc.
        df_copy['parsed_year'] = df_copy[year_col].astype(str).str.extract(r'(\d{4})')[0].astype(float)
    else:
        df_copy['parsed_year'] = df_copy[year_col]
    
    return df_copy

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
    <h1 class="nav-title">🏛️ Government Budget Analytics Platform</h1>
    <p class="nav-subtitle">Union Budget of India | 2014-2025 | Transparency · Analysis · Insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Budget Dataset (CSV)",
        type=['csv'],
        help="Upload a CSV file containing budget data"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.divider()
    
    # Theme Toggle
    theme = st.radio("🎨 Theme", ["Light", "Dark"], index=0)
    st.session_state.theme = theme.lower()

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🏢 Ministry Analysis",
    "🗺️ State Analysis",
    "📈 Forecasting",
    "🤖 AI Insights",
    "💾 Downloads"
])

# Check if data is loaded
if st.session_state.df is None:
    st.info("👆 Please upload a budget dataset CSV file using the sidebar to get started.")
    st.markdown("""
    ### 📋 Expected Dataset Format
    Your CSV should contain:
    - **Year Column**: e.g., "Year", "FY", "2016-17"
    - **Category Columns**: e.g., "Ministry", "Department", "State"
    - **Expenditure Columns**: Numeric values (BE, RE, AE, Actuals, etc.)
    
    The platform will automatically detect column types!
    """)
    st.stop()

# Load and parse data
df = st.session_state.df.copy()

# Detect columns
year_col = detect_year_column(df)
if year_col is None:
    st.error("❌ Could not detect a year column. Please ensure your dataset has a year/FY column.")
    st.stop()

categories = detect_category_columns(df)
df = parse_year_column(df, year_col)

# Get numeric columns
exclude_cols = [year_col, 'parsed_year'] + [v for v in categories.values() if v]
numeric_cols = detect_numeric_columns(df, exclude_cols)

if not numeric_cols:
    st.error("❌ No numeric expenditure columns found in the dataset.")
    st.stop()

# TAB 1: OVERVIEW
with tab1:
    st.markdown('<h2 class="section-header">📊 Budget Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Year range selector
    col1, col2 = st.columns(2)
    with col1:
        min_year = int(df['parsed_year'].min())
        max_year = int(df['parsed_year'].max())
        year_range = st.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    
    with col2:
        selected_metric = st.selectbox("Select Expenditure Metric", numeric_cols)
    
    # Filter data
    filtered_df = df[(df['parsed_year'] >= year_range[0]) & (df['parsed_year'] <= year_range[1])]
    
    # Calculate KPIs
    total_expenditure = filtered_df[selected_metric].sum()
    yearly_totals = filtered_df.groupby('parsed_year')[selected_metric].sum().sort_index()
    
    if len(yearly_totals) > 1:
        yoy_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2] * 100)
    else:
        yoy_growth = 0
    
    avg_expenditure = yearly_totals.mean()
    
    # Find top category
    if categories['ministry']:
        top_category = filtered_df.groupby(categories['ministry'])[selected_metric].sum().idxmax()
        top_category_value = filtered_df.groupby(categories['ministry'])[selected_metric].sum().max()
    else:
        top_category = "N/A"
        top_category_value = 0
    
    # Display KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Expenditure</div>
            <div class="kpi-value">₹{total_expenditure/1e5:.2f}L Cr</div>
            <div class="kpi-change positive">📅 {year_range[0]}-{year_range[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        change_class = "positive" if yoy_growth >= 0 else "negative"
        arrow = "📈" if yoy_growth >= 0 else "📉"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">YoY Growth</div>
            <div class="kpi-value">{yoy_growth:.2f}%</div>
            <div class="kpi-change {change_class}">{arrow} Year-over-Year</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Annual</div>
            <div class="kpi-value">₹{avg_expenditure/1e5:.2f}L Cr</div>
            <div class="kpi-change positive">📊 Mean Value</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Category</div>
            <div class="kpi-value" style="font-size: 1.2rem;">{top_category[:20]}</div>
            <div class="kpi-change positive">💰 ₹{top_category_value/1e5:.2f}L Cr</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Yearly Trend Chart
    st.markdown('<h3 class="section-header">📈 Yearly Expenditure Trend</h3>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_totals.index,
        y=yearly_totals.values,
        mode='lines+markers',
        name=selected_metric,
        line=dict(color='#004E89', width=3),
        marker=dict(size=10, color='#FF6B35'),
        fill='tozeroy',
        fillcolor='rgba(0, 78, 137, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Total {selected_metric} Over Time",
        xaxis_title="Year",
        yaxis_title=f"{selected_metric} (₹ Crores)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI-Generated Insights
    st.markdown('<h3 class="section-header">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    growth_trend = "increasing" if yoy_growth > 0 else "decreasing"
    
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">📊 Automated Analysis</div>
        <div class="insight-text">
            • The total expenditure from {year_range[0]} to {year_range[1]} is <strong>₹{total_expenditure/1e5:.2f} Lakh Crores</strong>.<br>
            • Year-over-year growth is <strong>{yoy_growth:.2f}%</strong>, indicating a <strong>{growth_trend}</strong> trend.<br>
            • The highest spending category is <strong>{top_category}</strong> with ₹{top_category_value/1e5:.2f}L Cr.<br>
            • Average annual expenditure during this period: <strong>₹{avg_expenditure/1e5:.2f}L Cr</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top Categories Table
    if categories['ministry']:
        st.markdown('<h3 class="section-header">🏆 Top 10 Ministries by Expenditure</h3>', unsafe_allow_html=True)
        
        top_ministries = filtered_df.groupby(categories['ministry'])[selected_metric].sum().sort_values(ascending=False).head(10)
        
        fig_bar = px.bar(
            x=top_ministries.values,
            y=top_ministries.index,
            orientation='h',
            title=f"Top 10 Ministries - {selected_metric}",
            labels={'x': f'{selected_metric} (₹ Crores)', 'y': 'Ministry'},
            color=top_ministries.values,
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig_bar, use_container_width=True)

# TAB 2: MINISTRY ANALYSIS
with tab2:
    st.markdown('<h2 class="section-header">🏢 Ministry Deep Dive</h2>', unsafe_allow_html=True)
    
    if categories['ministry']:
        ministries = sorted(df[categories['ministry']].unique())
        selected_ministry = st.selectbox("Select Ministry", ministries)
        
        ministry_df = df[df[categories['ministry']] == selected_ministry]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📊 {selected_ministry}")
            st.markdown(f"**Total Records:** {len(ministry_df)}")
            
            ministry_yearly = ministry_df.groupby('parsed_year')[numeric_cols].sum()
            
            fig = px.line(
                ministry_yearly,
                title=f"{selected_ministry} - Yearly Trends",
                labels={'value': 'Amount (₹ Crores)', 'parsed_year': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Latest Year Breakdown")
            latest_year = ministry_df['parsed_year'].max()
            latest_data = ministry_df[ministry_df['parsed_year'] == latest_year]
            
            if categories['department'] and categories['department'] in latest_data.columns:
                dept_totals = latest_data.groupby(categories['department'])[selected_metric].sum().sort_values(ascending=False).head(5)
                
                fig_pie = px.pie(
                    values=dept_totals.values,
                    names=dept_totals.index,
                    title=f"Top Departments in {int(latest_year)}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # BE vs RE vs AE Comparison
        budget_cols = [col for col in numeric_cols if any(x in col.upper() for x in ['BE', 'RE', 'AE'])]
        if len(budget_cols) >= 2:
            st.markdown('<h3 class="section-header">💰 Budget Estimates Comparison</h3>', unsafe_allow_html=True)
            
            comparison_data = ministry_yearly[budget_cols]
            
            fig_compare = go.Figure()
            for col in budget_cols:
                fig_compare.add_trace(go.Bar(name=col, x=comparison_data.index, y=comparison_data[col]))
            
            fig_compare.update_layout(barmode='group', title=f"{selected_ministry} - BE vs RE vs AE")
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Full table
        st.markdown('<h3 class="section-header">📋 Complete Data Table</h3>', unsafe_allow_html=True)
        st.dataframe(ministry_df, use_container_width=True, height=400)
    else:
        st.warning("No ministry column detected in the dataset.")

# TAB 3: STATE ANALYSIS
with tab3:
    st.markdown('<h2 class="section-header">🗺️ State-Level Analysis</h2>', unsafe_allow_html=True)
    
    if categories['state']:
        state_totals = filtered_df.groupby(categories['state'])[selected_metric].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_map = px.bar(
                x=state_totals.values,
                y=state_totals.index,
                orientation='h',
                title="State-wise Budget Allocation",
                labels={'x': f'{selected_metric} (₹ Crores)', 'y': 'State'},
                color=state_totals.values,
                color_continuous_scale='Viridis'
            )
            fig_map.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_map, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top 10 States")
            top_states_df = pd.DataFrame({
                'State': state_totals.head(10).index,
                'Amount (₹ Cr)': state_totals.head(10).values
            })
            st.dataframe(top_states_df, hide_index=True, use_container_width=True)
    else:
        st.info("No state column detected in this dataset.")

# TAB 4: FORECASTING
with tab4:
    st.markdown('<h2 class="section-header">📈 Budget Forecasting (Next 5 Years)</h2>', unsafe_allow_html=True)
    
    forecast_metric = st.selectbox("Select metric to forecast", numeric_cols, key='forecast_metric')
    
    # Prepare data for forecasting
    yearly_data = df.groupby('parsed_year')[forecast_metric].sum().reset_index()
    yearly_data = yearly_data.sort_values('parsed_year')
    
    X = yearly_data['parsed_year'].values.reshape(-1, 1)
    y = yearly_data[forecast_metric].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Forecast next 5 years
    last_year = int(yearly_data['parsed_year'].max())
    future_years = np.array([last_year + i for i in range(1, 6)]).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=yearly_data['parsed_year'],
            y=yearly_data[forecast_metric],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#004E89', width=3),
            marker=dict(size=10)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_years.flatten(),
            y=predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#FF6B35', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f"{forecast_metric} - 5-Year Forecast",
            xaxis_title="Year",
            yaxis_title=f"{forecast_metric} (₹ Crores)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Forecast Results")
        forecast_df = pd.DataFrame({
            'Year': future_years.flatten(),
            f'Predicted {forecast_metric}': predictions
        })
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)
        
        growth_rate = ((predictions[-1] - y[-1]) / y[-1] * 100)
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🎯 Projection Summary</div>
            <div class="insight-text">
                Expected growth over next 5 years: <strong>{growth_rate:.2f}%</strong><br>
                Projected {last_year + 5} value: <strong>₹{predictions[-1]/1e5:.2f}L Cr</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# TAB 5: AI INSIGHTS
with tab5:
    st.markdown('<h2 class="section-header">🤖 AI-Powered Query Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Ask questions about the budget data in natural language. The system will analyze and provide insights.
    """)
    
    query = st.text_input("💬 Ask a question:", placeholder="e.g., Which ministry grew the most after 2020?")
    
    if st.button("🔍 Analyze", type="primary"):
        if query:
            query_lower = query.lower()
            
            # Rule-based responses
            if 'grow' in query_lower or 'increase' in query_lower:
                if categories['ministry']:
                    growth_df = df.groupby([categories['ministry'], 'parsed_year'])[selected_metric].sum().unstack()
                    
                    if '2020' in query_lower:
                        pre_2020 = growth_df.loc[:, growth_df.columns < 2020].mean(axis=1)
                        post_2020 = growth_df.loc[:, growth_df.columns >= 2020].mean(axis=1)
                        growth = ((post_2020 - pre_2020) / pre_2020 * 100).sort_values(ascending=False)
                        
                        st.success(f"🎯 **Answer:** {growth.index[0]} showed the highest growth after 2020 with {growth.iloc[0]:.2f}% increase.")
                        
                        fig = px.bar(
                            x=growth.head(10).values,
                            y=growth.head(10).index,
                            orientation='h',
                            title="Top 10 Ministries by Growth (Post-2020)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            elif 'highest' in query_lower or 'most' in query_lower:
                if categories['ministry']:
                    totals = df.groupby(categories['ministry'])[selected_metric].sum().sort_values(ascending=False)
                    
                    st.success(f"🎯 **Answer:** {totals.index[0]} has the highest expenditure with ₹{totals.iloc[0]/1e5:.2f} Lakh Crores.")
                    
                    fig = px.pie(
                        values=totals.head(10).values,
                        names=totals.head(10).index,
                        title="Top 10 Ministries by Total Expenditure"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("💡 Try questions like: 'Which sector grew the most?', 'What has highest spending?', 'Show trends after 2020'")
        else:
            st.warning("Please enter a question.")

# TAB 6: DOWNLOADS
with tab6:
    st.markdown('<h2 class="section-header">💾 Download Center</h2>', unsafe_allow_html=True)
    
    st.markdown("Export your filtered and processed budget data in multiple formats.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 Filtered CSV")
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"budget_data_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### 📊 Excel Format")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Budget Data')
        
        st.download_button(
            label="⬇️ Download Excel",
            data=buffer.getvalue(),
            file_name=f"budget_data_{year_range[0]}_{year_range[1]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        st.markdown("### 📈 Yearly Summary")
        yearly_summary = filtered_df.groupby('parsed_year')[numeric_cols].sum().reset_index()
        csv_summary = yearly_summary.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Download Summary",
            data=csv_summary,
            file_name=f"yearly_summary_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
    
    st.divider()
    
    st.markdown("### 📋 Data Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

# Footer
st
