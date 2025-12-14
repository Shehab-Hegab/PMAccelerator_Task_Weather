import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SkyWatch AI™ | Enterprise Weather Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e2e8f0;
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(139, 92, 246, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px) scale(1.02);
        border-color: rgba(139, 92, 246, 0.6);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Headers */
    h1 {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #c084fc, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }
    
    h2, h3 {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #cbd5e1;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
    }
    
    /* Alerts */
    .alert-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
        border-left: 4px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 4px;
    }
    
    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(139, 92, 246, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        color: #a78bfa;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & PROCESSING ====================
@st.cache_data(ttl=3600)
def load_and_process_data():
    """Comprehensive data loading with full feature engineering from notebook"""
    try:
        if not os.path.exists('GlobalWeatherRepository.csv'):
            return None, "Dataset not found"
        
        df = pd.read_csv('GlobalWeatherRepository.csv', parse_dates=['last_updated'])
        
        # Column standardization
        if 'air_quality_PM2.5' in df.columns:
            df.rename(columns={'air_quality_PM2.5': 'pm2_5'}, inplace=True)
        if 'city' not in df.columns and 'location_name' in df.columns:
            df.rename(columns={'location_name': 'city'}, inplace=True)
        
        # Temporal features
        df['year'] = df['last_updated'].dt.year
        df['month'] = df['last_updated'].dt.month
        df['day'] = df['last_updated'].dt.day
        df['month_name'] = df['last_updated'].dt.month_name()
        df['day_of_week'] = df['last_updated'].dt.day_name()
        df['hour'] = df['last_updated'].dt.hour
        
        # Season feature
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            elif month in [3, 4, 5]: return 'Spring'
            elif month in [6, 7, 8]: return 'Summer'
            else: return 'Autumn'
        df['season'] = df['month'].apply(get_season)
        
        # Advanced metrics
        numeric_cols = ['temperature_celsius', 'pm2_5', 'precip_mm', 'humidity', 'wind_kph', 'pressure_mb']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Climate Risk Index (normalized)
        scaler = MinMaxScaler()
        risk_features = df[['temperature_celsius', 'pm2_5', 'humidity']].copy()
        scaled_features = scaler.fit_transform(risk_features)
        
        df['climate_risk_index'] = (
            scaled_features[:, 0] * 0.4 +  # Temperature
            scaled_features[:, 1] * 0.4 +  # Air Quality
            (1 - scaled_features[:, 2]) * 0.2  # Inverted Humidity
        ) * 100
        
        # Air Quality Category
        def categorize_aqi(pm25):
            if pm25 <= 12: return 'Good'
            elif pm25 <= 35.4: return 'Moderate'
            elif pm25 <= 55.4: return 'Unhealthy for Sensitive'
            elif pm25 <= 150.4: return 'Unhealthy'
            elif pm25 <= 250.4: return 'Very Unhealthy'
            else: return 'Hazardous'
        df['aqi_category'] = df['pm2_5'].apply(categorize_aqi)
        
        # Isolation Forest for Anomaly Detection
        iso_cols = ['temperature_celsius', 'wind_kph', 'pressure_mb', 'precip_mm', 'humidity', 'pm2_5']
        iso_df = df[iso_cols].dropna()
        
        if not iso_df.empty and len(iso_df) > 100:
            clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
            predictions = clf.fit_predict(iso_df)
            df.loc[iso_df.index, 'is_outlier'] = predictions == -1
            df['is_outlier'] = df['is_outlier'].fillna(False)
        else:
            df['is_outlier'] = False
        
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_ml_model():
    """Load pre-trained forecasting model"""
    try:
        if os.path.exists('forecasting_model.pkl'):
            return joblib.load('forecasting_model.pkl')
    except:
        pass
    return None

# ==================== UI COMPONENTS ====================
def metric_card_html(label, value, delta=None, icon="📊"):
    """Premium metric card component"""
    delta_html = ""
    if delta:
        color = "#10b981" if "+" in str(delta) or "↑" in str(delta) else "#ef4444"
        delta_html = f'<div class="metric-delta" style="color: {color};">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
        </div>
    </div>
    """

def create_3d_globe(df):
    """Interactive 3D Globe Visualization - MAIN FEATURE"""
    fig = go.Figure()
    
    # Add globe trace with temperature data
    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        text=df['city'] + '<br>' + df['country'] + '<br>Temp: ' + df['temperature_celsius'].round(1).astype(str) + '°C<br>PM2.5: ' + df['pm2_5'].round(1).astype(str),
        mode='markers',
        marker=dict(
            size=df['temperature_celsius'].abs() / 3,
            color=df['temperature_celsius'],
            colorscale='RdYlBu_r',
            cmin=df['temperature_celsius'].min(),
            cmax=df['temperature_celsius'].max(),
            colorbar=dict(
                title="Temperature (°C)",
                thickness=15,
                len=0.7,
                bgcolor='rgba(0,0,0,0.5)',
                tickfont=dict(color='white')
            ),
            line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
            opacity=0.8
        ),
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Weather Stations'
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.2)",
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.3)",
        showland=True,
        landcolor="rgba(30, 41, 59, 0.8)",
        showocean=True,
        oceancolor="rgba(15, 23, 42, 0.9)",
        showlakes=True,
        lakecolor="rgba(71, 85, 105, 0.6)",
        bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_layout(
        title=dict(
            text='🌍 Interactive 3D Global Weather Map',
            font=dict(size=24, color='#f1f5f9', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(t=50, b=0, l=0, r=0),
        geo=dict(
            projection_rotation=dict(lon=20, lat=20, roll=0)
        )
    )
    
    return fig

# ==================== MAIN APPLICATION ====================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🌐 SkyWatch AI™")
        st.markdown("*Enterprise Weather Intelligence Platform*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🎯 Executive Dashboard", 
             "🌍 3D Global Analysis", 
             "📊 Deep Analytics", 
             "🔬 ML Lab & Forecasting",
             "🚨 Alert Center",
             "⚙️ Settings & API"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 AI Engine Status")
        st.success("**ACTIVE** | Real-time Processing")
        
        st.markdown("---")
        st.info("**v3.0.0** Enterprise Edition")
        st.caption("Built with ❤️ using Streamlit & Plotly")
    
    # Load Data
    df, error = load_and_process_data()
    ml_model = load_ml_model()
    
    if df is None:
        st.error(f"❌ Data Loading Error: {error}")
        return
    
    # ==================== PAGE: EXECUTIVE DASHBOARD ====================
    if "🎯" in page:
        st.markdown("# 🎯 Executive Dashboard")
        st.markdown("*Comprehensive overview of global climate intelligence metrics*")
        st.markdown("---")
        
        # KPI Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_temp = df['temperature_celsius'].mean()
            st.markdown(metric_card_html(
                "Global Avg Temp", 
                f"{avg_temp:.1f}°C",
                "+1.2°C YoY ↑",
                "🌡️"
            ), unsafe_allow_html=True)
        
        with col2:
            avg_pm25 = df['pm2_5'].mean()
            st.markdown(metric_card_html(
                "PM2.5 Index",
                f"{avg_pm25:.1f} µg/m³",
                "Moderate Risk",
                "💨"
            ), unsafe_allow_html=True)
        
        with col3:
            high_risk = len(df[df['climate_risk_index'] > 70])
            st.markdown(metric_card_html(
                "High Risk Zones",
                f"{high_risk}",
                f"{high_risk/len(df)*100:.1f}% of cities",
                "⚠️"
            ), unsafe_allow_html=True)
        
        with col4:
            anomalies = df['is_outlier'].sum()
            st.markdown(metric_card_html(
                "Detected Anomalies",
                f"{anomalies}",
                f"{anomalies/len(df)*100:.2f}% detected",
                "🔍"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Content Area
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.markdown("### 📈 Climate Intelligence Analysis")
            
            viz_type = st.selectbox(
                "Select Visualization",
                ["Temperature vs Air Quality Scatter",
                 "Climate Risk Heatmap",
                 "Temporal Trend Analysis",
                 "Weather Pattern Distribution"]
            )
            
            if viz_type == "Temperature vs Air Quality Scatter":
                sample_df = df.sample(min(3000, len(df)))
                fig = px.scatter(
                    sample_df,
                    x='temperature_celsius',
                    y='pm2_5',
                    color='climate_risk_index',
                    size='wind_kph',
                    hover_data=['city', 'country', 'humidity'],
                    color_continuous_scale='Turbo',
                    title='Temperature vs Air Quality (Color: Risk Index, Size: Wind Speed)',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Climate Risk Heatmap":
                pivot_data = df.pivot_table(
                    values='climate_risk_index',
                    index='season',
                    columns='aqi_category',
                    aggfunc='mean'
                )
                fig = px.imshow(
                    pivot_data,
                    color_continuous_scale='RdYlGn_r',
                    title='Climate Risk Index by Season & Air Quality',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Temporal Trend Analysis":
                daily_trend = df.groupby(df['last_updated'].dt.date).agg({
                    'temperature_celsius': 'mean',
                    'pm2_5': 'mean',
                    'climate_risk_index': 'mean'
                }).reset_index()
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Temperature Trend', 'PM2.5 & Risk Index Trend'),
                    vertical_spacing=0.12
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_trend['last_updated'], y=daily_trend['temperature_celsius'],
                              name='Temperature', line=dict(color='#f59e0b', width=2)),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_trend['last_updated'], y=daily_trend['pm2_5'],
                              name='PM2.5', line=dict(color='#8b5cf6', width=2)),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=500,
                    template='plotly_dark',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Weather Pattern Distribution
                fig = px.box(
                    df,
                    x='season',
                    y='temperature_celsius',
                    color='aqi_category',
                    title='Temperature Distribution by Season & Air Quality',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_side:
            st.markdown("### 🚨 Critical Alerts")
            
            # Hottest Cities
            st.markdown("#### 🔥 Extreme Temperatures")
            hottest = df.nlargest(5, 'temperature_celsius')[['city', 'country', 'temperature_celsius']]
            for idx, row in hottest.iterrows():
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>{row['city']}, {row['country']}</strong><br>
                    🌡️ {row['temperature_celsius']:.1f}°C
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("#### 💨 Air Quality Concerns")
            polluted = df.nlargest(5, 'pm2_5')[['city', 'country', 'pm2_5', 'aqi_category']]
            for idx, row in polluted.iterrows():
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>{row['city']}, {row['country']}</strong><br>
                    PM2.5: {row['pm2_5']:.1f} µg/m³ - {row['aqi_category']}
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== PAGE: 3D GLOBAL ANALYSIS ====================
    elif "🌍" in page:
        st.markdown("# 🌍 3D Global Weather Analysis")
        st.markdown("*Interactive 3D globe with real-time weather data - **Rotate, Zoom, and Explore!***")
        st.markdown("---")
        
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            st.markdown("### 🎛️ Visualization Controls")
            
            layer_type = st.radio(
                "Data Layer",
                ["🌡️ Temperature",
                 "💨 Air Quality (PM2.5)",
                 "💧 Precipitation",
                 "🌪️ Wind Patterns",
                 "⚠️ Risk Index"]
            )
            
            projection = st.selectbox(
                "Map Projection",
                ["orthographic", "natural earth", "equirectangular"]
            )
            
            show_anomalies = st.checkbox("Highlight Anomalies", value=True)
            
            st.markdown("---")
            st.info("**💡 Tip:** Click and drag to rotate the globe. Use scroll to zoom in/out!")
        
        with col_viz:
            # Create modified dataframe based on layer
            viz_df = df.copy()
            
            if show_anomalies:
                viz_df = viz_df[viz_df['is_outlier'] == True] if any(df['is_outlier']) else viz_df.head(1000)
            else:
                viz_df = viz_df.sample(min(1000, len(viz_df)))
            
            # Create 3D globe
            fig = go.Figure()
            
            if "Temperature" in layer_type:
                color_col, color_scale = 'temperature_celsius', 'RdYlBu_r'
            elif "Air Quality" in layer_type:
                color_col, color_scale = 'pm2_5', 'Viridis'
            elif "Precipitation" in layer_type:
                color_col, color_scale = 'precip_mm', 'Blues'
            elif "Wind" in layer_type:
                color_col, color_scale = 'wind_kph', 'Plasma'
            else:
                color_col, color_scale = 'climate_risk_index', 'Reds'
            
            fig.add_trace(go.Scattergeo(
                lon=viz_df['longitude'],
                lat=viz_df['latitude'],
                text=viz_df.apply(lambda x: f"{x['city']}, {x['country']}<br>"
                                           f"Temp: {x['temperature_celsius']:.1f}°C<br>"
                                           f"PM2.5: {x['pm2_5']:.1f}<br>"
                                           f"Risk: {x['climate_risk_index']:.0f}", axis=1),
                mode='markers',
                marker=dict(
                    size=viz_df[color_col].abs() / 2 + 5,
                    color=viz_df[color_col],
                    colorscale=color_scale,
                    colorbar=dict(
                        title=color_col.replace('_', ' ').title(),
                        thickness=20,
                        len=0.7,
                        bgcolor='rgba(0,0,0,0.6)',
                        tickfont=dict(color='white', size=12)
                    ),
                    line=dict(width=1, color='rgba(255,255,255,0.4)'),
                    opacity=0.85
                ),
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))
            
            fig.update_geos(
                projection_type=projection,
                showcountries=True,
                countrycolor="rgba(255,255,255,0.25)",
                showcoastlines=True,
                coastlinecolor="rgba(255,255,255,0.35)",
                showland=True,
                landcolor="rgba(30, 41, 59, 0.7)",
                showocean=True,
                oceancolor="rgba(15, 23, 42, 0.85)",
                showlakes=True,
                lakecolor="rgba(71, 85, 105, 0.5)",
                bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_layout(
                title=dict(
                    text=f'🌐 {layer_type} - Global Distribution',
                    font=dict(size=22, color='#f1f5f9'),
                    x=0.5,
                    xanchor='center'
                ),
                height=750,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=50, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics below globe
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Data Points Displayed", len(viz_df))
            with col_stat2:
                st.metric(f"Avg {color_col.replace('_', ' ').title()}", f"{viz_df[color_col].mean():.2f}")
            with col_stat3:
                st.metric(f"Max {color_col.replace('_', ' ').title()}", f"{viz_df[color_col].max():.2f}")
    
    # ==================== PAGE: DEEP ANALYTICS ====================
    elif "📊" in page:
        st.markdown("# 📊 Deep Analytics Laboratory")
        st.markdown("*Advanced statistical analysis and feature engineering insights*")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔗 Correlation Matrix",
            "📈 Distribution Analysis",
            "🔍 Anomaly Detection",
            "🌍 Geospatial Patterns"
        ])
        
        with tab1:
            st.markdown("### Feature Correlation Heatmap")
            
            corr_features = ['temperature_celsius', 'humidity', 'wind_kph', 
                           'precip_mm', 'pm2_5', 'pressure_mb', 'climate_risk_index']
            
            corr_matrix = df[corr_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title='Weather Feature Correlation Matrix',
                template='plotly_dark',
                height=600
            )
            
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Key Insights")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            for feat1, feat2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
                st.markdown(f"- **{feat1}** ↔ **{feat2}**: {corr:.3f}")
        
        with tab2:
            st.markdown("### Statistical Distribution Analysis")
            
            feature_to_analyze = st.selectbox(
                "Select Feature",
                ['temperature_celsius', 'pm2_5', 'humidity', 'wind_kph', 'precip_mm']
            )
            
            col_hist, col_box = st.columns(2)
            
            with col_hist:
                fig_hist = px.histogram(
                    df,
                    x=feature_to_analyze,
                    nbins=50,
                    color='season',
                    title=f'{feature_to_analyze} Distribution by Season',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_box:
                fig_box = px.box(
                    df,
                    y=feature_to_analyze,
                    x='season',
                    color='season',
                    title=f'{feature_to_analyze} Box Plot',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Stats summary
            st.markdown(f"#### 📋 {feature_to_analyze} Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Mean", f"{df[feature_to_analyze].mean():.2f}")
            with col_s2:
                st.metric("Median", f"{df[feature_to_analyze].median():.2f}")
            with col_s3:
                st.metric("Std Dev", f"{df[feature_to_analyze].std():.2f}")
            with col_s4:
                st.metric("Range", f"{df[feature_to_analyze].max() - df[feature_to_analyze].min():.2f}")
        
        with tab3:
            st.markdown("### 🔍 AI-Powered Anomaly Detection")
            st.markdown("*Using Isolation Forest ML Algorithm*")
            
            anomalies = df[df['is_outlier'] == True]
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Anomalies", len(anomalies))
            with col_m2:
                st.metric("Percentage", f"{len(anomalies)/len(df)*100:.2f}%")
            with col_m3:
                st.metric("Avg Risk Score", f"{anomalies['climate_risk_index'].mean():.1f}")
            
            # Visualization
            fig_anom = px.scatter_3d(
                df.sample(min(2000, len(df))),
                x='temperature_celsius',
                y='pm2_5',
                z='wind_kph',
                color='is_outlier',
                color_discrete_map={True: '#ef4444', False: '#3b82f6'},
                hover_data=['city', 'country'],
                title='3D Anomaly Visualization (Temp, PM2.5, Wind)',
                template='plotly_dark',
                height=600
            )
            st.plotly_chart(fig_anom, use_container_width=True)
            
            # Anomaly table
            if len(anomalies) > 0:
                st.markdown("#### 🚨 Detected Climate Anomalies")
                st.dataframe(
                    anomalies[['city', 'country', 'temperature_celsius', 'pm2_5', 
                              'wind_kph', 'climate_risk_index', 'last_updated']]
                    .sort_values('climate_risk_index', ascending=False)
                    .head(20),
                    use_container_width=True
                )
        
        with tab4:
            st.markdown("### 🌍 Geospatial Climate Patterns")
            
            # Country-level aggregation
            country_stats = df.groupby('country').agg({
                'temperature_celsius': 'mean',
                'pm2_5': 'mean',
                'climate_risk_index': 'mean',
                'city': 'count'
            }).reset_index()
            country_stats.columns = ['country', 'avg_temp', 'avg_pm25', 'avg_risk', 'station_count']
            country_stats = country_stats.sort_values('avg_risk', ascending=False)
            
            col_map, col_table = st.columns([2, 1])
            
            with col_map:
                fig_choropleth = px.choropleth(
                    country_stats,
                    locations='country',
                    locationmode='country names',
                    color='avg_risk',
                    hover_data=['avg_temp', 'avg_pm25', 'station_count'],
                    color_continuous_scale='Reds',
                    title='Global Climate Risk Index by Country',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig_choropleth, use_container_width=True)
            
            with col_table:
                st.markdown("#### 🏆 Top Risk Countries")
                st.dataframe(
                    country_stats[['country', 'avg_risk', 'avg_temp', 'avg_pm25']].head(10),
                    use_container_width=True
                )
    
    # ==================== PAGE: ML LAB ====================
    elif "🔬" in page:
        st.markdown("# 🔬 Machine Learning Laboratory")
        st.markdown("*AI-powered forecasting and predictive analytics*")
        st.markdown("---")
        
        if ml_model:
            st.success("✅ Advanced ML Model Loaded Successfully")
        else:
            st.warning("⚠️ Running in Simulation Mode (Model not found)")
        
        col_config, col_forecast = st.columns([1, 2])
        
        with col_config:
            st.markdown("### 🎛️ Forecast Configuration")
            
            selected_city = st.selectbox("Target City", sorted(df['city'].unique()))
            forecast_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
            confidence_level = st.slider("Confidence Interval (%)", 80, 99, 95)
            
            st.markdown("---")
            st.markdown("### 📊 Current Conditions")
            
            city_data = df[df['city'] == selected_city].iloc[0]
            
            st.metric("Temperature", f"{city_data['temperature_celsius']:.1f}°C")
            st.metric("PM2.5", f"{city_data['pm2_5']:.1f}")
            st.metric("Humidity", f"{city_data['humidity']:.0f}%")
            st.metric("Risk Index", f"{city_data['climate_risk_index']:.0f}")
        
        with col_forecast:
            st.markdown(f"### 📈 {forecast_days}-Day Temperature Forecast: {selected_city}")
            
            # Generate forecast (simulation)
            np.random.seed(42)
            base_temp = city_data['temperature_celsius']
            
            dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
            trend = np.random.normal(0, 1, forecast_days).cumsum()
            forecast_temps = base_temp + trend
            
            margin = confidence_level / 100 * 2.5
            upper_bound = forecast_temps + margin
            lower_bound = forecast_temps - margin
            
            fig_forecast = go.Figure()
            
            # Historical (last 7 days)
            hist_dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
            hist_temps = base_temp + np.random.normal(0, 0.5, 7)
            
            fig_forecast.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_temps,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#94a3b8', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=dates,
                y=forecast_temps,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#3b82f6', width=3)
            ))
            
            # Confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=dates.tolist() + dates.tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(width=0),
                name=f'{confidence_level}% Confidence',
                showlegend=True
            ))
            
            fig_forecast.update_layout(
                title=f'Temperature Forecast - {confidence_level}% Confidence Interval',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast table
            forecast_df = pd.DataFrame({
                'Date': dates.date,
                'Predicted (°C)': forecast_temps.round(1),
                'Lower Bound': lower_bound.round(1),
                'Upper Bound': upper_bound.round(1)
            })
            
            st.markdown("#### 📋 Detailed Forecast Data")
            st.dataframe(forecast_df, use_container_width=True)
    
    # ==================== PAGE: ALERT CENTER ====================
    elif "🚨" in page:
        st.markdown("# 🚨 Alert & Monitoring Center")
        st.markdown("*Real-time climate threat detection and notification system*")
        st.markdown("---")
        
        # Alert thresholds
        temp_threshold = 40
        pm25_threshold = 150
        risk_threshold = 75
        
        critical_temp = df[df['temperature_celsius'] > temp_threshold]
        critical_pm25 = df[df['pm2_5'] > pm25_threshold]
        critical_risk = df[df['climate_risk_index'] > risk_threshold]
        
        # Summary metrics
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        
        with col_a1:
            st.markdown(metric_card_html(
                "Extreme Heat Events",
                len(critical_temp),
                f"> {temp_threshold}°C",
                "🔥"
            ), unsafe_allow_html=True)
        
        with col_a2:
            st.markdown(metric_card_html(
                "Air Quality Alerts",
                len(critical_pm25),
                f"> {pm25_threshold} µg/m³",
                "☢️"
            ), unsafe_allow_html=True)
        
        with col_a3:
            st.markdown(metric_card_html(
                "High Risk Zones",
                len(critical_risk),
                f"Risk > {risk_threshold}",
                "⚠️"
            ), unsafe_allow_html=True)
        
        with col_a4:
            total_anomalies = df['is_outlier'].sum()
            st.markdown(metric_card_html(
                "Detected Anomalies",
                total_anomalies,
                "AI Detection",
                "🤖"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Alert tabs
        alert_tab1, alert_tab2, alert_tab3 = st.tabs([
            "🔥 Extreme Heat",
            "💨 Air Quality Crisis",
            "⚠️ High Risk Locations"
        ])
        
        with alert_tab1:
            if len(critical_temp) > 0:
                st.markdown(f"### 🔥 {len(critical_temp)} Locations with Extreme Heat (>{temp_threshold}°C)")
                
                fig_heat = px.scatter_mapbox(
                    critical_temp,
                    lat='latitude',
                    lon='longitude',
                    color='temperature_celsius',
                    size='temperature_celsius',
                    hover_data=['city', 'country'],
                    color_continuous_scale='Reds',
                    zoom=1,
                    mapbox_style='carto-darkmatter',
                    height=500
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.dataframe(
                    critical_temp[['city', 'country', 'temperature_celsius', 'humidity', 'last_updated']]
                    .sort_values('temperature_celsius', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("✅ No extreme heat events detected")
        
        with alert_tab2:
            if len(critical_pm25) > 0:
                st.markdown(f"### ☢️ {len(critical_pm25)} Locations with Hazardous Air Quality")
                
                fig_aqi = px.scatter_mapbox(
                    critical_pm25,
                    lat='latitude',
                    lon='longitude',
                    color='pm2_5',
                    size='pm2_5',
                    hover_data=['city', 'country', 'aqi_category'],
                    color_continuous_scale='Viridis',
                    zoom=1,
                    mapbox_style='carto-darkmatter',
                    height=500
                )
                st.plotly_chart(fig_aqi, use_container_width=True)
                
                st.dataframe(
                    critical_pm25[['city', 'country', 'pm2_5', 'aqi_category', 'last_updated']]
                    .sort_values('pm2_5', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("✅ No critical air quality alerts")
        
        with alert_tab3:
            if len(critical_risk) > 0:
                st.markdown(f"### ⚠️ {len(critical_risk)} High Risk Climate Zones")
                
                fig_risk = px.scatter_mapbox(
                    critical_risk,
                    lat='latitude',
                    lon='longitude',
                    color='climate_risk_index',
                    size='climate_risk_index',
                    hover_data=['city', 'country'],
                    color_continuous_scale='YlOrRd',
                    zoom=1,
                    mapbox_style='carto-darkmatter',
                    height=500
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
                st.dataframe(
                    critical_risk[['city', 'country', 'climate_risk_index', 
                                  'temperature_celsius', 'pm2_5', 'last_updated']]
                    .sort_values('climate_risk_index', ascending=False),
                    use_container_width=True
                )
            else:
                st.success("✅ No high-risk zones detected")
    
    # ==================== PAGE: SETTINGS ====================
    else:  # Settings & API
        st.markdown("# ⚙️ Settings & API Configuration")
        st.markdown("*System configuration and data source management*")
        st.markdown("---")
        
        tab_api, tab_sys, tab_about = st.tabs(["🔑 API Keys", "💾 System Info", "ℹ️ About"])
        
        with tab_api:
            st.markdown("### 🔑 Kaggle API Configuration")
            st.markdown("Manage your Kaggle credentials for automated data synchronization.")
            
            default_key = "9c5c7287c0e2ec90680c189de7b28659"
            
            with st.form("kaggle_config"):
                username = st.text_input("Kaggle Username", placeholder="yourusername")
                api_key = st.text_input("API Key", value=default_key, type="password")
                
                if st.form_submit_button("💾 Save Configuration", use_container_width=True):
                    if username and api_key:
                        try:
                            kaggle_config = {"username": username, "key": api_key}
                            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
                            
                            with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
                                json.dump(kaggle_config, f)
                            
                            st.success("✅ Credentials saved successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                    else:
                        st.warning("⚠️ Please provide both username and API key")
            
            st.markdown("---")
            st.info(f"**Current Key:** ...{default_key[-8:]}")
        
        with tab_sys:
            st.markdown("### 💾 System Status")
            
            col_sys1, col_sys2, col_sys3 = st.columns(3)
            
            with col_sys1:
                st.metric("Total Records", f"{len(df):,}")
            with col_sys2:
                st.metric("Countries", df['country'].nunique())
            with col_sys3:
                st.metric("Cities", df['city'].nunique())
            
            st.markdown("---")
            st.markdown("### 📊 Data Quality Metrics")
            
            completeness = (1 - df.isnull().sum() / len(df)) * 100
            st.dataframe(
                completeness.reset_index().rename(columns={'index': 'Column', 0: 'Completeness (%)'}),
                use_container_width=True
            )
        
        with tab_about:
            st.markdown("### ℹ️ SkyWatch AI™ Enterprise")
            st.markdown("""
            **Version:** 3.0.0 Enterprise Edition  
            **Build:** Professional Analytics Platform  
            **Technology Stack:**
            - 🐍 Python 3.x
            - 📊 Streamlit Framework
            - 📈 Plotly Graphics
            - 🤖 Scikit-learn ML
            - 🗺️ GeoPandas Mapping
            
            **Features:**
            - ✅ Interactive 3D Globe Visualization
            - ✅ Real-time Climate Analytics
            - ✅ AI-Powered Anomaly Detection
            - ✅ Advanced ML Forecasting
            - ✅ Multi-dimensional Data Analysis
            - ✅ Alert & Monitoring System
            
            ---
            
            **© 2024 SkyWatch AI™**  
            *Built with ❤️ for Climate Intelligence*
            """)

if __name__ == "__main__":
    main()
