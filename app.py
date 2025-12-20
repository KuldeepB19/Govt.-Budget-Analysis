import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import re
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="India Budget Analytics 2014-2025",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 95, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .kpi-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .kpi-delta.positive {
        background: #dcfce7;
        color: #166534;
    }
    
    .kpi-delta.negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a5f;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #3d7ab5;
        margin: 2rem 0 1.5rem 0;
    }
    
    .insight-box {
        background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 5px solid #3d7ab5;
        padding: 1.25rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
    }
    
    .insight-box h4 {
        color: #1e3a5f;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .insight-box p {
        color: #334155;
        margin: 0;
        line-height: 1.6;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .download-btn:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-weight: 500;
        color: #374151;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
    }
    
    div[data-testid="stMetric"] label {
        color: #64748b;
        font-weight: 500;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1e3a5f;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_numeric(value):
    """Clean and convert string to numeric value."""
    if pd.isna(value) or value == '-' or value == '':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).replace(',', '').replace(' ', '').strip()
        return float(cleaned) if cleaned and cleaned != '-' else 0.0
    except:
        return 0.0


def extract_year(year_str):
    """Extract numeric year from string like '2014-15' or '2014-2015'."""
    if pd.isna(year_str):
        return None
    match = re.search(r'(\d{4})', str(year_str))
    return int(match.group(1)) if match else None


def format_currency(value, in_crores=True):
    """Format number as Indian currency."""
    if value >= 100000:
        return f"₹{value/100000:.2f}L Cr"
    elif value >= 1000:
        return f"₹{value/1000:.2f}K Cr"
    else:
        return f"₹{value:.2f} Cr"


def format_large_number(value):
    """Format large numbers with abbreviations."""
    if abs(value) >= 1e5:
        return f"{value/1e5:.2f}L"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return 0
    return ((end_value / start_value) ** (1 / years) - 1) * 100


def load_and_process_data(uploaded_file):
    """Load and process the budget CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Define numeric columns
        numeric_cols = [
            'Revenue (Plan)', 'Capital (Plan)', 'Total (Plan)',
            'Revenue (Non-Plan)', 'Capital (Non-Plan)', 'Total (Non-Plan)',
            'Total Plan & Non-Plan'
        ]
        
        # Clean numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        # Extract numeric year
        df['Numeric_Year'] = df['Year'].apply(extract_year)
        
        # Standardize ministry names
        df['Ministry Name'] = df['Ministry Name'].str.strip().str.upper()
        
        # Handle ministry name variations
        df['Ministry Name'] = df['Ministry Name'].replace({
            'MINISTRY OF AGRICULTURE': 'MINISTRY OF AGRICULTURE AND FARMERS WELFARE',
            "MINISTRY OF AGRICULTURE AND FARMERS' WELFARE": 'MINISTRY OF AGRICULTURE AND FARMERS WELFARE',
        })
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def generate_insights(df, year_range, budget_type):
    """Generate automatic insights from the data."""
    insights = []
    
    filtered = df[(df['Numeric_Year'] >= year_range[0]) & (df['Numeric_Year'] <= year_range[1])]
    
    if filtered.empty:
        return ["No data available for the selected period."]
    
    # Total budget trend
    yearly_total = filtered.groupby('Numeric_Year')[budget_type].sum()
    if len(yearly_total) > 1:
        first_year = yearly_total.iloc[0]
        last_year = yearly_total.iloc[-1]
        growth = ((last_year - first_year) / first_year) * 100 if first_year > 0 else 0
        direction = "increased" if growth > 0 else "decreased"
        insights.append(f"📈 Total budget {direction} by {abs(growth):.1f}% from {year_range[0]} to {year_range[1]}.")
    
    # Top growing ministry
    ministry_growth = filtered.groupby('Ministry Name').apply(
        lambda x: x[budget_type].iloc[-1] - x[budget_type].iloc[0] if len(x) > 1 else 0
    ).sort_values(ascending=False)
    
    if len(ministry_growth) > 0 and ministry_growth.iloc[0] > 0:
        top_grower = ministry_growth.index[0]
        growth_amt = ministry_growth.iloc[0]
        insights.append(f"🚀 {top_grower.title()} showed the highest absolute growth of ₹{format_large_number(growth_amt)} Cr.")
    
    # Budget concentration
    total_budget = filtered.groupby('Ministry Name')[budget_type].sum().sort_values(ascending=False)
    if len(total_budget) >= 2:
        top2_share = (total_budget.iloc[:2].sum() / total_budget.sum()) * 100
        insights.append(f"🎯 Top 2 ministries account for {top2_share:.1f}% of total allocations.")
    
    # YoY volatility
    yearly_totals = filtered.groupby('Numeric_Year')[budget_type].sum()
    if len(yearly_totals) > 2:
        yoy_changes = yearly_totals.pct_change().dropna() * 100
        max_change_year = yoy_changes.abs().idxmax()
        max_change = yoy_changes[max_change_year]
        direction = "increase" if max_change > 0 else "decrease"
        insights.append(f"📊 Highest YoY {direction} of {abs(max_change):.1f}% occurred in {int(max_change_year)}.")
    
    return insights


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇮🇳 India Union Budget Analytics</h1>
        <p>Comprehensive analysis of budget allocations from 2014 to 2025</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/255px-Flag_of_India.svg.png", width=80)
        st.markdown("### 📂 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Budget CSV",
            type=['csv'],
            help="Upload a CSV file with budget allocation data"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard provides comprehensive 
        analytics for India's Union Budget data.
        
        **Features:**
        - 📊 Overview Dashboard
        - 🏢 Ministry Analysis
        - 📈 ML Forecasting
        - 🤖 AI Query System
        - 💾 Data Downloads
        """)
        
        st.markdown("---")
        st.markdown("### 📧 Contact")
        st.markdown("Built with ❤️ using Streamlit")
    
    # Check if data is uploaded
    if uploaded_file is None:
        st.info("👆 Please upload a Budget CSV file using the sidebar to get started.")
        
        # Show sample format
        st.markdown("### 📋 Expected CSV Format")
        sample_data = {
            'Ministry Name': ['MINISTRY OF DEFENCE', 'MINISTRY OF FINANCE'],
            'Year': ['2014-2015', '2014-2015'],
            'Revenue (Plan)': [134440, 2397.74],
            'Capital (Plan)': [9739, 2.5],
            'Total (Plan)': [144179, 2400.24],
            'Revenue (Non-Plan)': [100778, 163013.91],
            'Capital (Non-Plan)': [66085, 557.08],
            'Total (Non-Plan)': [166863, 163570.99],
            'Total Plan & Non-Plan': [311042, 165971.23]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        return
    
    # Load and process data
    df = load_and_process_data(uploaded_file)
    
    if df is None:
        return
    
    # Store in session state
    st.session_state['budget_data'] = df
    
    # Get data info
    years = sorted(df['Numeric_Year'].dropna().unique())
    ministries = sorted(df['Ministry Name'].unique())
    min_year, max_year = int(min(years)), int(max(years))
    
    # Create tabs
    tabs = st.tabs([
        "📊 Overview Dashboard",
        "🏢 Ministry Analysis",
        "📈 Forecasting",
        "🤖 AI Insights",
        "💾 Download Center"
    ])
    
    # ============================================
    # TAB 1: OVERVIEW DASHBOARD
    # ============================================
    with tabs[0]:
        st.markdown('<p class="section-header">📊 Budget Overview Dashboard</p>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns([2, 1])
        with col1:
            year_range = st.slider(
                "Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="overview_year_range"
            )
        with col2:
            budget_type = st.selectbox(
                "Budget Type",
                ["Total Plan & Non-Plan", "Total (Plan)", "Total (Non-Plan)"],
                key="overview_budget_type"
            )
        
        # Filter data
        filtered_df = df[(df['Numeric_Year'] >= year_range[0]) & (df['Numeric_Year'] <= year_range[1])]
        
        # KPI Calculations
        total_allocation = filtered_df[budget_type].sum()
        yearly_totals = filtered_df.groupby('Numeric_Year')[budget_type].sum()
        
        if len(yearly_totals) >= 2:
            yoy_growth = ((yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2] * 100) if yearly_totals.iloc[-2] > 0 else 0
        else:
            yoy_growth = 0
        
        avg_allocation = yearly_totals.mean() if len(yearly_totals) > 0 else 0
        
        top_ministry = filtered_df.groupby('Ministry Name')[budget_type].sum().idxmax() if not filtered_df.empty else "N/A"
        top_ministry_value = filtered_df.groupby('Ministry Name')[budget_type].sum().max() if not filtered_df.empty else 0
        
        # KPI Cards
        st.markdown("### Key Performance Indicators")
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            st.metric(
                label="Total Allocation",
                value=format_currency(total_allocation),
                delta=f"{yoy_growth:+.1f}% YoY" if yoy_growth != 0 else None
            )
        
        with kpi_cols[1]:
            st.metric(
                label="YoY Growth",
                value=f"{yoy_growth:+.1f}%",
                delta="Latest vs Previous Year"
            )
        
        with kpi_cols[2]:
            st.metric(
                label="Avg Annual Allocation",
                value=format_currency(avg_allocation)
            )
        
        with kpi_cols[3]:
            st.metric(
                label="Top Ministry",
                value=top_ministry.replace("MINISTRY OF ", "").title()[:20] + "..." if len(top_ministry) > 25 else top_ministry.replace("MINISTRY OF ", "").title()
            )
        
        st.markdown("---")
        
        # Charts Row 1
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 📈 Yearly Budget Trend")
            yearly_data = filtered_df.groupby('Numeric_Year')[budget_type].sum().reset_index()
            fig_trend = px.line(
                yearly_data,
                x='Numeric_Year',
                y=budget_type,
                markers=True,
                template='plotly_white'
            )
            fig_trend.update_traces(
                line=dict(color='#1e3a5f', width=3),
                marker=dict(size=10, color='#3d7ab5')
            )
            fig_trend.update_layout(
                xaxis_title="Year",
                yaxis_title="Allocation (₹ Crores)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with chart_col2:
            st.markdown("#### 🏆 Top 10 Ministries")
            top_ministries = filtered_df.groupby('Ministry Name')[budget_type].sum().nlargest(10).reset_index()
            top_ministries['Ministry Short'] = top_ministries['Ministry Name'].str.replace('MINISTRY OF ', '').str.title()
            
            fig_bar = px.bar(
                top_ministries,
                x=budget_type,
                y='Ministry Short',
                orientation='h',
                template='plotly_white',
                color=budget_type,
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Allocation (₹ Crores)",
                yaxis_title=""
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Charts Row 2
        pie_col1, pie_col2 = st.columns(2)
        
        with pie_col1:
            st.markdown("#### 📊 Plan vs Non-Plan Distribution")
            plan_total = filtered_df['Total (Plan)'].sum()
            nonplan_total = filtered_df['Total (Non-Plan)'].sum()
            
            fig_pie1 = px.pie(
                values=[plan_total, nonplan_total],
                names=['Plan', 'Non-Plan'],
                template='plotly_white',
                color_discrete_sequence=['#1e3a5f', '#3d7ab5']
            )
            fig_pie1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie1, use_container_width=True)
        
        with pie_col2:
            st.markdown("#### 💰 Revenue vs Capital Expenditure")
            revenue_total = filtered_df['Revenue (Plan)'].sum() + filtered_df['Revenue (Non-Plan)'].sum()
            capital_total = filtered_df['Capital (Plan)'].sum() + filtered_df['Capital (Non-Plan)'].sum()
            
            fig_pie2 = px.pie(
                values=[revenue_total, capital_total],
                names=['Revenue', 'Capital'],
                template='plotly_white',
                color_discrete_sequence=['#059669', '#10b981']
            )
            fig_pie2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie2, use_container_width=True)
        
        # Insights Box
        st.markdown("---")
        insights = generate_insights(df, year_range, budget_type)
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Auto-Generated Insights</h4>
        </div>
        """, unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f"- {insight}")
    
    # ============================================
    # TAB 2: MINISTRY ANALYSIS
    # ============================================
    with tabs[1]:
        st.markdown('<p class="section-header">🏢 Ministry-wise Analysis</p>', unsafe_allow_html=True)
        
        # Ministry selector
        selected_ministry = st.selectbox(
            "Select Ministry",
            options=ministries,
            format_func=lambda x: x.replace("MINISTRY OF ", "").title()
        )
        
        ministry_df = df[df['Ministry Name'] == selected_ministry].sort_values('Numeric_Year')
        
        if ministry_df.empty:
            st.warning("No data available for selected ministry.")
        else:
            # Ministry KPIs
            st.markdown("### Ministry KPIs")
            m_cols = st.columns(4)
            
            with m_cols[0]:
                st.metric(
                    "Total Allocation",
                    format_currency(ministry_df['Total Plan & Non-Plan'].sum())
                )
            
            with m_cols[1]:
                st.metric(
                    "Average Annual",
                    format_currency(ministry_df['Total Plan & Non-Plan'].mean())
                )
            
            with m_cols[2]:
                st.metric(
                    "Years Covered",
                    f"{len(ministry_df)} years"
                )
            
            with m_cols[3]:
                latest = ministry_df[ministry_df['Numeric_Year'] == ministry_df['Numeric_Year'].max()]['Total Plan & Non-Plan'].values
                st.metric(
                    "Latest Allocation",
                    format_currency(latest[0]) if len(latest) > 0 else "N/A"
                )
            
            st.markdown("---")
            
            # Trend Chart
            st.markdown("#### 📈 Budget Trend Over Years")
            fig_ministry = go.Figure()
            
            fig_ministry.add_trace(go.Scatter(
                x=ministry_df['Numeric_Year'],
                y=ministry_df['Total (Plan)'],
                name='Plan',
                mode='lines+markers',
                line=dict(color='#1e3a5f', width=2)
            ))
            
            fig_ministry.add_trace(go.Scatter(
                x=ministry_df['Numeric_Year'],
                y=ministry_df['Total (Non-Plan)'],
                name='Non-Plan',
                mode='lines+markers',
                line=dict(color='#3d7ab5', width=2)
            ))
            
            fig_ministry.add_trace(go.Scatter(
                x=ministry_df['Numeric_Year'],
                y=ministry_df['Total Plan & Non-Plan'],
                name='Total',
                mode='lines+markers',
                line=dict(color='#059669', width=3)
            ))
            
            fig_ministry.update_layout(
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                xaxis_title="Year",
                yaxis_title="Allocation (₹ Crores)"
            )
            st.plotly_chart(fig_ministry, use_container_width=True)
            
            # Revenue vs Capital
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                st.markdown("#### 💵 Revenue vs Capital (Plan)")
                fig_rc = go.Figure()
                fig_rc.add_trace(go.Bar(
                    x=ministry_df['Numeric_Year'],
                    y=ministry_df['Revenue (Plan)'],
                    name='Revenue',
                    marker_color='#1e3a5f'
                ))
                fig_rc.add_trace(go.Bar(
                    x=ministry_df['Numeric_Year'],
                    y=ministry_df['Capital (Plan)'],
                    name='Capital',
                    marker_color='#3d7ab5'
                ))
                fig_rc.update_layout(
                    barmode='group',
                    template='plotly_white',
                    xaxis_title="Year",
                    yaxis_title="Allocation (₹ Crores)"
                )
                st.plotly_chart(fig_rc, use_container_width=True)
            
            with chart_cols[1]:
                st.markdown("#### 📊 YoY Growth Rate")
                ministry_df['YoY_Growth'] = ministry_df['Total Plan & Non-Plan'].pct_change() * 100
                
                colors = ['#059669' if x >= 0 else '#dc2626' for x in ministry_df['YoY_Growth'].fillna(0)]
                
                fig_yoy = go.Figure()
                fig_yoy.add_trace(go.Bar(
                    x=ministry_df['Numeric_Year'],
                    y=ministry_df['YoY_Growth'],
                    marker_color=colors
                ))
                fig_yoy.update_layout(
                    template='plotly_white',
                    xaxis_title="Year",
                    yaxis_title="Growth Rate (%)"
                )
                st.plotly_chart(fig_yoy, use_container_width=True)
            
            # Data Table
            st.markdown("#### 📋 Detailed Data Table")
            display_df = ministry_df[['Year', 'Revenue (Plan)', 'Capital (Plan)', 'Total (Plan)',
                                      'Revenue (Non-Plan)', 'Capital (Non-Plan)', 'Total (Non-Plan)',
                                      'Total Plan & Non-Plan']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ============================================
    # TAB 3: FORECASTING
    # ============================================
    with tabs[2]:
        st.markdown('<p class="section-header">📈 Budget Forecasting</p>', unsafe_allow_html=True)
        
        forecast_col1, forecast_col2 = st.columns(2)
        
        with forecast_col1:
            forecast_ministry = st.selectbox(
                "Select Ministry for Forecast",
                options=['All Ministries'] + list(ministries),
                format_func=lambda x: x.replace("MINISTRY OF ", "").title() if x != 'All Ministries' else x
            )
        
        with forecast_col2:
            forecast_metric = st.selectbox(
                "Select Metric",
                ["Total Plan & Non-Plan", "Total (Plan)", "Total (Non-Plan)"]
            )
        
        # Prepare data for forecasting
        if forecast_ministry == 'All Ministries':
            forecast_data = df.groupby('Numeric_Year')[forecast_metric].sum().reset_index()
        else:
            forecast_data = df[df['Ministry Name'] == forecast_ministry].groupby('Numeric_Year')[forecast_metric].sum().reset_index()
        
        forecast_data = forecast_data.dropna()
        
        if len(forecast_data) < 3:
            st.warning("Not enough data points for forecasting. Need at least 3 years of data.")
        else:
            # Train model
            X = forecast_data['Numeric_Year'].values.reshape(-1, 1)
            y = forecast_data[forecast_metric].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Future predictions
            future_years = np.array([max_year + i for i in range(1, 6)]).reshape(-1, 1)
            future_pred = model.predict(future_years)
            
            # Calculate confidence interval (simple approach)
            residuals = y - y_pred
            std_error = np.std(residuals)
            
            # Create forecast chart
            st.markdown("#### 🔮 Historical Data & Forecast")
            
            fig_forecast = go.Figure()
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['Numeric_Year'],
                y=forecast_data[forecast_metric],
                name='Historical',
                mode='lines+markers',
                line=dict(color='#1e3a5f', width=3),
                marker=dict(size=10)
            ))
            
            # Fitted line
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['Numeric_Year'],
                y=y_pred,
                name='Trend Line',
                mode='lines',
                line=dict(color='#3d7ab5', width=2, dash='dash')
            ))
            
            # Future predictions
            fig_forecast.add_trace(go.Scatter(
                x=future_years.flatten(),
                y=future_pred,
                name='Forecast',
                mode='lines+markers',
                line=dict(color='#059669', width=3),
                marker=dict(size=10, symbol='diamond')
            ))
            
            # Confidence band
            fig_forecast.add_trace(go.Scatter(
                x=np.concatenate([future_years.flatten(), future_years.flatten()[::-1]]),
                y=np.concatenate([future_pred + 1.96 * std_error, (future_pred - 1.96 * std_error)[::-1]]),
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence'
            ))
            
            fig_forecast.update_layout(
                template='plotly_white',
                xaxis_title="Year",
                yaxis_title="Allocation (₹ Crores)",
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            st.markdown("#### 📊 Forecast Metrics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                total_growth = ((future_pred[-1] - y[-1]) / y[-1] * 100) if y[-1] > 0 else 0
                st.metric("Projected 5-Year Growth", f"{total_growth:+.1f}%")
            
            with metric_cols[1]:
                cagr = calculate_cagr(y[-1], future_pred[-1], 5)
                st.metric("Projected CAGR", f"{cagr:.2f}%")
            
            with metric_cols[2]:
                st.metric("Model R² Score", f"{r2:.4f}")
            
            with metric_cols[3]:
                st.metric("Trend Slope", f"₹{model.coef_[0]:,.0f} Cr/year")
            
            # Forecast table
            st.markdown("#### 📋 Forecast Details")
            forecast_table = pd.DataFrame({
                'Year': future_years.flatten(),
                'Predicted Allocation (₹ Cr)': future_pred.round(2),
                'Lower Bound (₹ Cr)': (future_pred - 1.96 * std_error).round(2),
                'Upper Bound (₹ Cr)': (future_pred + 1.96 * std_error).round(2)
            })
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    
    # ============================================
    # TAB 4: AI INSIGHTS (Rule-Based NLP)
    # ============================================
    with tabs[3]:
        st.markdown('<p class="section-header">🤖 AI-Powered Query System</p>', unsafe_allow_html=True)
        
        st.markdown("""
        Ask questions about the budget data in natural language. Try queries like:
        - "Which ministry grew the most after 2018?"
        - "Top 5 ministries by budget"
        - "Trend for Defence budget"
        - "Compare Plan vs Non-Plan"
        - "Total budget in 2020"
        """)
        
        user_query = st.text_input("🔍 Enter your question:", placeholder="e.g., Which ministry has the highest budget?")
        
        if user_query:
            query_lower = user_query.lower()
            
            # Intent detection
            response_text = ""
            show_chart = False
            chart_data = None
            chart_type = None
            
            # Pattern: Top N ministries
            if re.search(r'top\s*(\d+)', query_lower) or 'highest' in query_lower or 'largest' in query_lower:
                match = re.search(r'top\s*(\d+)', query_lower)
                n = int(match.group(1)) if match else 5
                
                year_match = re.search(r'(20\d{2})', query_lower)
                if year_match:
                    year = int(year_match.group(1))
                    year_df = df[df['Numeric_Year'] == year]
                    top_n = year_df.groupby('Ministry Name')['Total Plan & Non-Plan'].sum().nlargest(n)
                    response_text = f"**Top {n} Ministries by Budget in {year}:**\n"
                else:
                    top_n = df.groupby('Ministry Name')['Total Plan & Non-Plan'].sum().nlargest(n)
                    response_text = f"**Top {n} Ministries by Total Budget (All Years):**\n"
                
                for i, (ministry, value) in enumerate(top_n.items(), 1):
                    response_text += f"\n{i}. {ministry.replace('MINISTRY OF ', '').title()}: {format_currency(value)}"
                
                chart_data = top_n.reset_index()
                chart_data.columns = ['Ministry', 'Budget']
                chart_type = 'bar'
                show_chart = True
            
            # Pattern: Growth analysis
            elif 'grew' in query_lower or 'growth' in query_lower or 'increased' in query_lower:
                year_match = re.search(r'after\s*(20\d{2})', query_lower)
                start_year = int(year_match.group(1)) if year_match else min_year
                
                filtered = df[df['Numeric_Year'] >= start_year]
                
                growth_data = filtered.groupby('Ministry Name').apply(
                    lambda x: ((x['Total Plan & Non-Plan'].iloc[-1] - x['Total Plan & Non-Plan'].iloc[0]) / 
                              x['Total Plan & Non-Plan'].iloc[0] * 100) if len(x) > 1 and x['Total Plan & Non-Plan'].iloc[0] > 0 else 0
                ).sort_values(ascending=False)
                
                top_grower = growth_data.index[0]
                top_growth = growth_data.iloc[0]
                
                response_text = f"**Ministry with Highest Growth since {start_year}:**\n\n"
                response_text += f"🏆 {top_grower.replace('MINISTRY OF ', '').title()} with **{top_growth:.1f}%** growth\n\n"
                response_text += "**Top 5 by Growth Rate:**\n"
                
                for ministry, growth in growth_data.head(5).items():
                    response_text += f"\n- {ministry.replace('MINISTRY OF ', '').title()}: {growth:+.1f}%"
                
                chart_data = growth_data.head(10).reset_index()
                chart_data.columns = ['Ministry', 'Growth %']
                chart_type = 'bar'
                show_chart = True
            
            # Pattern: Trend for specific ministry
            elif 'trend' in query_lower:
                for ministry in ministries:
                    if any(word in query_lower for word in ministry.lower().split()):
                        ministry_df = df[df['Ministry Name'] == ministry].sort_values('Numeric_Year')
                        response_text = f"**Budget Trend for {ministry.replace('MINISTRY OF ', '').title()}:**\n\n"
                        
                        if len(ministry_df) > 1:
                            first_val = ministry_df['Total Plan & Non-Plan'].iloc[0]
                            last_val = ministry_df['Total Plan & Non-Plan'].iloc[-1]
                            growth = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                            
                            response_text += f"- First Year ({ministry_df['Numeric_Year'].iloc[0]}): {format_currency(first_val)}\n"
                            response_text += f"- Latest Year ({ministry_df['Numeric_Year'].iloc[-1]}): {format_currency(last_val)}\n"
                            response_text += f"- Overall Growth: **{growth:+.1f}%**"
                        
                        chart_data = ministry_df[['Numeric_Year', 'Total Plan & Non-Plan']].copy()
                        chart_data.columns = ['Year', 'Budget']
                        chart_type = 'line'
                        show_chart = True
                        break
            
            # Pattern: Compare Plan vs Non-Plan
            elif 'plan' in query_lower and 'non' in query_lower:
                plan_total = df['Total (Plan)'].sum()
                nonplan_total = df['Total (Non-Plan)'].sum()
                
                response_text = f"**Plan vs Non-Plan Comparison:**\n\n"
                response_text += f"- Total Plan Budget: {format_currency(plan_total)}\n"
                response_text += f"- Total Non-Plan Budget: {format_currency(nonplan_total)}\n"
                response_text += f"- Plan Share: **{(plan_total/(plan_total+nonplan_total)*100):.1f}%**\n"
                response_text += f"- Non-Plan Share: **{(nonplan_total/(plan_total+nonplan_total)*100):.1f}%**"
                
                chart_data = pd.DataFrame({
                    'Type': ['Plan', 'Non-Plan'],
                    'Amount': [plan_total, nonplan_total]
                })
                chart_type = 'pie'
                show_chart = True
            
            # Pattern: Total budget in specific year
            elif re.search(r'(20\d{2})', query_lower) and ('total' in query_lower or 'budget' in query_lower):
                year_match = re.search(r'(20\d{2})', query_lower)
                year = int(year_match.group(1))
                
                year_df = df[df['Numeric_Year'] == year]
                total = year_df['Total Plan & Non-Plan'].sum()
                
                response_text = f"**Total Budget in {year}:**\n\n"
                response_text += f"💰 {format_currency(total)}\n\n"
                response_text += f"**Ministry-wise Breakdown:**\n"
                
                ministry_breakdown = year_df.groupby('Ministry Name')['Total Plan & Non-Plan'].sum().sort_values(ascending=False)
                for ministry, value in ministry_breakdown.head(5).items():
                    response_text += f"\n- {ministry.replace('MINISTRY OF ', '').title()}: {format_currency(value)}"
                
                chart_data = ministry_breakdown.head(10).reset_index()
                chart_data.columns = ['Ministry', 'Budget']
                chart_type = 'bar'
                show_chart = True
            
            else:
                response_text = """
                I couldn't understand your query. Try asking:
                - "Top 5 ministries by budget"
                - "Which ministry grew the most after 2018?"
                - "Trend for Defence budget"
                - "Compare Plan vs Non-Plan"
                - "Total budget in 2020"
                """
            
            # Display response
            st.markdown(response_text)
            
            # Display chart if applicable
            if show_chart and chart_data is not None:
                st.markdown("---")
                if chart_type == 'bar':
                    chart_data['Short Name'] = chart_data.iloc[:, 0].str.replace('MINISTRY OF ', '').str.title()
                    fig = px.bar(
                        chart_data,
                        x=chart_data.columns[1],
                        y='Short Name',
                        orientation='h',
                        template='plotly_white',
                        color=chart_data.columns[1],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                elif chart_type == 'line':
                    fig = px.line(
                        chart_data,
                        x='Year',
                        y='Budget',
                        markers=True,
                        template='plotly_white'
                    )
                    fig.update_traces(line=dict(color='#1e3a5f', width=3))
                elif chart_type == 'pie':
                    fig = px.pie(
                        chart_data,
                        values='Amount',
                        names='Type',
                        template='plotly_white',
                        color_discrete_sequence=['#1e3a5f', '#3d7ab5']
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 5: DOWNLOAD CENTER
    # ============================================
    with tabs[4]:
        st.markdown('<p class="section-header">💾 Download Center</p>', unsafe_allow_html=True)
        
        # Filters for download
        st.markdown("### Filter Data for Download")
        
        dl_col1, dl_col2 = st.columns(2)
        
        with dl_col1:
            dl_year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="download_year_range"
            )
        
        with dl_col2:
            dl_ministries = st.multiselect(
                "Select Ministries",
                options=ministries,
                default=ministries[:3],
                format_func=lambda x: x.replace("MINISTRY OF ", "").title()
            )
        
        # Filter data
        filtered_download = df[
            (df['Numeric_Year'] >= dl_year_range[0]) &
            (df['Numeric_Year'] <= dl_year_range[1]) &
            (df['Ministry Name'].isin(dl_ministries) if dl_ministries else True)
        ]
        
        # Preview
        st.markdown("### Data Preview")
        st.dataframe(filtered_download.head(20), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 20 of {len(filtered_download)} rows")
        
        st.markdown("---")
        
        # Download buttons
        st.markdown("### Download Options")
        
        download_cols = st.columns(4)
        
        with download_cols[0]:
            csv_data = filtered_download.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="budget_data_filtered.csv",
                mime="text/csv"
            )
        
        with download_cols[1]:
            buffer = BytesIO()
            filtered_download.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name="budget_data_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with download_cols[2]:
            yearly_summary = filtered_download.groupby('Numeric_Year').agg({
                'Total (Plan)': 'sum',
                'Total (Non-Plan)': 'sum',
                'Total Plan & Non-Plan': 'sum'
            }).reset_index()
            yearly_csv = yearly_summary.to_csv(index=False)
            st.download_button(
                label="📥 Yearly Summary",
                data=yearly_csv,
                file_name="yearly_summary.csv",
                mime="text/csv"
            )
        
        with download_cols[3]:
            ministry_summary = filtered_download.groupby('Ministry Name').agg({
                'Total (Plan)': 'sum',
                'Total (Non-Plan)': 'sum',
                'Total Plan & Non-Plan': 'sum'
            }).reset_index()
            ministry_csv = ministry_summary.to_csv(index=False)
            st.download_button(
                label="📥 Ministry Summary",
                data=ministry_csv,
                file_name="ministry_summary.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🇮🇳 India Union Budget Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>Data Source: Government of India Budget Documents (2014-2025)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
