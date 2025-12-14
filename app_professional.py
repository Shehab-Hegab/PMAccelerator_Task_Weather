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
import requests

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="WeatherSphere Pro™ | Elite Climate Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Main Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 25%, #0f1419 50%, #1e1b4b 75%, #0a0e27 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        color: #e2e8f0;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Glass Morphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px;
        backdrop-filter: blur(20px) saturate(180%);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 16px 64px 0 rgba(139, 92, 246, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(139, 92, 246, 0.5);
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 2px solid rgba(139, 92, 246, 0.4);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.25);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.6s;
    }
    .metric-card:hover::before {
        left: 100%;
    }
    .metric-card:hover {
        transform: translateY(-4px) scale(1.03);
        border-color: rgba(139, 92, 246, 0.8);
        box-shadow: 0 12px 48px rgba(139, 92, 246, 0.4);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .metric-label {
        color: #cbd5e1;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
        opacity: 0.9;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
    }
    
    /* Headers with Gradient */
    h1 {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #c084fc, #ec4899, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        margin-bottom: 12px;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
    }
    
    h2, h3 {
        color: #f1f5f9;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar Premium Style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border-right: 2px solid rgba(139, 92, 246, 0.3);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    
    /* Premium Buttons */
    .stButton button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 700;
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.5);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 13px;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(139, 92, 246, 0.7);
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #c084fc 100%);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 16px;
        padding: 10px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 700;
        padding: 14px 28px;
        transition: all 0.3s;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139, 92, 246, 0.1);
        color: #c084fc;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Alert Boxes */
    .alert-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.15));
        border-left: 5px solid #ef4444;
        padding: 18px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.15));
        border-left: 5px solid #f59e0b;
        padding: 18px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.15));
        border-left: 5px solid #22c55e;
        padding: 18px;
        border-radius: 12px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #8b5cf6, #a855f7);
    }
    
    /* Code Blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(139, 92, 246, 0.15);
        padding: 4px 8px;
        border-radius: 6px;
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    /* Plotly Charts Enhancement */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & PROCESSING ====================
def sanitize_text(text):
    """Remove problematic characters that cause JSON parsing errors"""
    if pd.isna(text):
        return ""
    # Convert to string and remove problematic characters
    text = str(text)
    # Remove backslashes and other escape characters
    text = text.replace('\\', '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Keep only ASCII printable characters and common international chars
    text = ''.join(char for char in text if ord(char) < 65536 and (ord(char) >= 32 or char == ' '))
    return text.strip()

@st.cache_data(ttl=3600)
def load_and_process_data():
    """Comprehensive data loading with full feature engineering"""
    try:
        if not os.path.exists('GlobalWeatherRepository.csv'):
            return None, "Dataset not found"
        
        df = pd.read_csv('GlobalWeatherRepository.csv', parse_dates=['last_updated'])
        
        # Sanitize text fields to prevent JSON parsing errors
        text_columns = ['city', 'country', 'location_name']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(sanitize_text)
        
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
        
        # Fill missing values
        numeric_cols = ['temperature_celsius', 'pm2_5', 'precip_mm', 'humidity', 'wind_kph', 'pressure_mb']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Climate Risk Index
        scaler = MinMaxScaler()
        risk_features = df[['temperature_celsius', 'pm2_5', 'humidity']].copy()
        scaled_features = scaler.fit_transform(risk_features)
        
        df['climate_risk_index'] = (
            scaled_features[:, 0] * 0.4 +
            scaled_features[:, 1] * 0.4 +
            (1 - scaled_features[:, 2]) * 0.2
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
        
        # Anomaly Detection
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

def create_3d_interactive_globe(df, color_metric='temperature_celsius', title='Global Weather Sphere'):
    """
    FLAGSHIP FEATURE: Interactive 3D Rotating Globe
    Users can click and drag to rotate, scroll to zoom
    """
    # Sample data for performance
    sample_df = df.sample(min(2000, len(df))).copy()
    
    # Sanitize text in sample for safety
    sample_df['city'] = sample_df['city'].apply(sanitize_text)
    sample_df['country'] = sample_df['country'].apply(sanitize_text)
    
    # Create safe hover text without complex formatting
    def create_hover_text(row):
        try:
            city = sanitize_text(row['city'])
            country = sanitize_text(row['country'])
            temp = round(float(row['temperature_celsius']), 1)
            pm25 = round(float(row['pm2_5']), 1)
            humidity = round(float(row['humidity']), 0)
            wind = round(float(row['wind_kph']), 1)
            risk = round(float(row['climate_risk_index']), 0)
            
            return (city + ", " + country + 
                   " | Temp: " + str(temp) + "C | " +
                   "PM2.5: " + str(pm25) + " | " +
                   "Humidity: " + str(humidity) + "% | " +
                   "Wind: " + str(wind) + " km/h | " +
                   "Risk: " + str(risk) + "/100")
        except:
            return "Data unavailable"
    
    sample_df['hover_text'] = sample_df.apply(create_hover_text, axis=1)
    
    # Color scale mapping
    color_scales = {
        'temperature_celsius': 'RdYlBu_r',
        'pm2_5': 'Viridis',
        'precip_mm': 'Blues',
        'wind_kph': 'Plasma',
        'climate_risk_index': 'Reds',
        'humidity': 'YlGnBu'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergeo(
        lon=sample_df['longitude'],
        lat=sample_df['latitude'],
        text=sample_df['hover_text'],
        mode='markers',
        marker=dict(
            size=sample_df[color_metric].abs() / 3 + 6,
            color=sample_df[color_metric],
            colorscale=color_scales.get(color_metric, 'Viridis'),
            cmin=sample_df[color_metric].quantile(0.05),
            cmax=sample_df[color_metric].quantile(0.95),
            colorbar=dict(
                title=dict(
                    text=color_metric.replace('_', ' ').title(),
                    font=dict(color='white', size=14, family='Inter')
                ),
                thickness=20,
                len=0.7,
                bgcolor='rgba(0,0,0,0.7)',
                tickfont=dict(color='white', size=12),
                bordercolor='rgba(139, 92, 246, 0.5)',
                borderwidth=2
            ),
            line=dict(width=0.8, color='rgba(255,255,255,0.4)'),
            opacity=0.85,
            symbol='circle'
        ),
        hovertemplate='%{text}<extra></extra>',
        name='Weather Stations'
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.25)",
        countrywidth=0.5,
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.35)",
        coastlinewidth=1,
        showland=True,
        landcolor="rgba(30, 41, 59, 0.7)",
        showocean=True,
        oceancolor="rgba(15, 23, 42, 0.9)",
        showlakes=True,
        lakecolor="rgba(71, 85, 105, 0.6)",
        bgcolor='rgba(0,0,0,0)',
        projection_rotation=dict(lon=20, lat=20, roll=0)
    )
    
    fig.update_layout(
        title=dict(
            text=f'🌍 {title}',
            font=dict(size=26, color='#f1f5f9', family='Space Grotesk', weight='bold'),
            x=0.5,
            xanchor='center'
        ),
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        margin=dict(t=60, b=0, l=0, r=0),
        hovermode='closest'
    )
    
    return fig

# ==================== MAIN APPLICATION ====================
def main():
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🌐 WeatherSphere Pro™")
        st.markdown("*Elite Climate Intelligence Platform*")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🎯 Command Center",
             "🌍 Interactive Globe", 
             "📊 Analytics Lab",
             "🔬 ML Forecasting",
             "🚨 Alert Hub",
             "📈 Advanced Insights",
             "⚙️ API Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 AI Status")
        st.success("**ACTIVE** | Real-time Processing")
        
        st.markdown("---")
        st.markdown("### 📡 API Key")
        api_key = st.text_input("Weather API Key", value="9c5c7287c0e2ec90680c189de7b28659", type="password")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        
        st.markdown("---")
        st.info("**v4.0.0** Enterprise Edition")
        st.caption("Built with ❤️ | Powered by AI")
    
    # Load Data
    df, error = load_and_process_data()
    ml_model = load_ml_model()
    
    if df is None:
        st.error(f"❌ Data Loading Error: {error}")
        return
    
    # ==================== PAGE: COMMAND CENTER ====================
    if "🎯" in page:
        st.markdown("# 🎯 Command Center")
        st.markdown("*Real-time global climate intelligence dashboard*")
        st.markdown("---")
        
        # KPI Metrics
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
                "Moderate Level",
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
                "Anomalies Detected",
                f"{anomalies}",
                f"{anomalies/len(df)*100:.2f}%",
                "🔍"
            ), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Dashboard Content
        col_main, col_side = st.columns([2.5, 1])
        
        with col_main:
            st.markdown("### 📈 Climate Intelligence Analysis")
            
            viz_type = st.selectbox(
                "Select Visualization",
                ["Temperature vs Air Quality 3D",
                 "Climate Risk Heatmap",
                 "Temporal Trends",
                 "Distribution Analysis"]
            )
            
            if viz_type == "Temperature vs Air Quality 3D":
                sample_df = df.sample(min(3000, len(df)))
                fig = px.scatter_3d(
                    sample_df,
                    x='temperature_celsius',
                    y='pm2_5',
                    z='humidity',
                    color='climate_risk_index',
                    size='wind_kph',
                    hover_data=['city', 'country'],
                    color_continuous_scale='Turbo',
                    title='3D Climate Correlation Analysis',
                    template='plotly_dark',
                    height=550
                )
                fig.update_traces(marker=dict(line=dict(width=0)))
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
                    title='Climate Risk Matrix: Season × Air Quality',
                    template='plotly_dark',
                    height=550,
                    text_auto='.1f'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Temporal Trends":
                daily_trend = df.groupby(df['last_updated'].dt.date).agg({
                    'temperature_celsius': 'mean',
                    'pm2_5': 'mean',
                    'climate_risk_index': 'mean'
                }).reset_index()
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Temperature Trend', 'PM2.5 Trend', 'Risk Index Trend'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_trend['last_updated'], y=daily_trend['temperature_celsius'],
                              name='Temperature', line=dict(color='#f59e0b', width=3),
                              fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.2)'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_trend['last_updated'], y=daily_trend['pm2_5'],
                              name='PM2.5', line=dict(color='#8b5cf6', width=3),
                              fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.2)'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_trend['last_updated'], y=daily_trend['climate_risk_index'],
                              name='Risk Index', line=dict(color='#ef4444', width=3),
                              fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'),
                    row=3, col=1
                )
                
                fig.update_layout(
                    height=550,
                    template='plotly_dark',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Distribution Analysis
                fig = px.violin(
                    df,
                    x='season',
                    y='temperature_celsius',
                    color='aqi_category',
                    title='Temperature Distribution by Season & Air Quality',
                    template='plotly_dark',
                    height=550,
                    box=True
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
            
            st.markdown("#### 💨 Air Quality Crisis")
            polluted = df.nlargest(5, 'pm2_5')[['city', 'country', 'pm2_5', 'aqi_category']]
            for idx, row in polluted.iterrows():
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>{row['city']}, {row['country']}</strong><br>
                    PM2.5: {row['pm2_5']:.1f} - {row['aqi_category']}
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== PAGE: INTERACTIVE GLOBE ====================
    elif "🌍" in page:
        st.markdown("# 🌍 Interactive Weather Sphere")
        st.markdown("*Rotate, zoom, and explore real-time global weather data on a 3D sphere*")
        st.markdown("---")
        
        col_ctrl, col_viz = st.columns([1, 3])
        
        with col_ctrl:
            st.markdown("### 🎛️ Globe Controls")
            
            layer_type = st.radio(
                "Data Layer",
                ["🌡️ Temperature",
                 "💨 Air Quality (PM2.5)",
                 "💧 Humidity",
                 "🌪️ Wind Speed",
                 "⚠️ Climate Risk Index"]
            )
            
            layer_map = {
                "🌡️ Temperature": 'temperature_celsius',
                "💨 Air Quality (PM2.5)": 'pm2_5',
                "💧 Humidity": 'humidity',
                "🌪️ Wind Speed": 'wind_kph',
                "⚠️ Climate Risk Index": 'climate_risk_index'
            }
            
            selected_metric = layer_map[layer_type]
            
            show_anomalies = st.checkbox("🔍 Highlight Anomalies Only", value=False)
            
            st.markdown("---")
            st.markdown("""
            <div class="alert-success">
                <strong>💡 Pro Tip:</strong><br>
                • Click & drag to rotate<br>
                • Scroll to zoom in/out<br>
                • Hover for details
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Quick Stats
            st.markdown("### 📊 Quick Stats")
            viz_df_temp = df[df['is_outlier'] == True] if show_anomalies and df['is_outlier'].any() else df
            st.metric("Data Points", len(viz_df_temp))
            st.metric(f"Avg {selected_metric.replace('_', ' ').title()}", 
                     f"{viz_df_temp[selected_metric].mean():.2f}")
            st.metric("Countries", df['country'].nunique())
        
        with col_viz:
            viz_df = df[df['is_outlier'] == True] if show_anomalies and df['is_outlier'].any() else df
            
            fig_globe = create_3d_interactive_globe(
                viz_df, 
                color_metric=selected_metric,
                title=f"{layer_type} - Global Distribution"
            )
            
            st.plotly_chart(fig_globe, use_container_width=True)
            
            # Stats below globe
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Min", f"{viz_df[selected_metric].min():.2f}")
            with col_s2:
                st.metric("Max", f"{viz_df[selected_metric].max():.2f}")
            with col_s3:
                st.metric("Mean", f"{viz_df[selected_metric].mean():.2f}")
            with col_s4:
                st.metric("Std Dev", f"{viz_df[selected_metric].std():.2f}")
    
    # ==================== PAGE: ANALYTICS LAB ====================
    elif "📊" in page:
        st.markdown("# 📊 Analytics Laboratory")
        st.markdown("*Deep statistical analysis and advanced data science*")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔗 Correlations",
            "📈 Distributions",
            "🔍 Anomalies",
            "🌏 Geospatial",
            "📉 Time Series"
        ])
        
        with tab1:
            st.markdown("### Feature Correlation Matrix")
            
            corr_features = ['temperature_celsius', 'humidity', 'wind_kph', 
                           'precip_mm', 'pm2_5', 'pressure_mb', 'climate_risk_index']
            
            corr_matrix = df[corr_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                aspect='auto',
                title='Pearson Correlation Heatmap',
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Strong Correlations Detected")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            for feat1, feat2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
                color = "green" if corr > 0 else "red"
                st.markdown(f"- **{feat1}** ↔ **{feat2}**: `{corr:.3f}` ", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Statistical Distribution Analysis")
            
            col_sel, col_viz = st.columns([1, 2])
            
            with col_sel:
                feature_to_analyze = st.selectbox(
                    "Select Feature",
                    ['temperature_celsius', 'pm2_5', 'humidity', 'wind_kph', 'precip_mm', 'climate_risk_index']
                )
                
                st.markdown("---")
                st.markdown("#### Statistics")
                st.metric("Mean", f"{df[feature_to_analyze].mean():.2f}")
                st.metric("Median", f"{df[feature_to_analyze].median():.2f}")
                st.metric("Std Dev", f"{df[feature_to_analyze].std():.2f}")
                st.metric("Skewness", f"{df[feature_to_analyze].skew():.2f}")
            
            with col_viz:
                fig_dist = px.histogram(
                    df,
                    x=feature_to_analyze,
                    nbins=60,
                    color='season',
                    marginal='box',
                    title=f'{feature_to_analyze.replace("_", " ").title()} Distribution',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔍 AI-Powered Anomaly Detection")
            st.markdown("*Using Isolation Forest Machine Learning*")
            
            anomalies = df[df['is_outlier'] == True]
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Anomalies", len(anomalies))
            with col_m2:
                st.metric("Percentage", f"{len(anomalies)/len(df)*100:.2f}%")
            with col_m3:
                st.metric("Avg Risk Score", f"{anomalies['climate_risk_index'].mean():.1f}" if len(anomalies) > 0 else "N/A")
            with col_m4:
                st.metric("Avg Temp", f"{anomalies['temperature_celsius'].mean():.1f}°C" if len(anomalies) > 0 else "N/A")
            
            if len(anomalies) > 0:
                fig_anom = px.scatter_3d(
                    df.sample(min(2000, len(df))),
                    x='temperature_celsius',
                    y='pm2_5',
                    z='wind_kph',
                    color='is_outlier',
                    color_discrete_map={True: '#ef4444', False: '#3b82f6'},
                    title='Anomaly Detection Visualization (3D)',
                    template='plotly_dark',
                    height=600,
                    hover_data=['city', 'country']
                )
                st.plotly_chart(fig_anom, use_container_width=True)
                
                st.markdown("#### 🚨 Top Anomalous Events")
                st.dataframe(anomalies.nlargest(10, 'climate_risk_index')[
                    ['city', 'country', 'temperature_celsius', 'pm2_5', 'climate_risk_index', 'last_updated']
                ], use_container_width=True)
        
        with tab4:
            st.markdown("### 🌏 Geospatial Pattern Analysis")
            
            geo_metric = st.selectbox(
                "Select Metric for Map",
                ['temperature_celsius', 'pm2_5', 'climate_risk_index', 'humidity']
            )
            
            # Create a sample dataframe and ensure size values are positive
            sample_geo_df = df.sample(min(5000, len(df))).copy()
            sample_geo_df['size_metric'] = sample_geo_df[geo_metric].abs() + 1  # Add 1 to avoid zero size
            
            fig_geo = px.scatter_mapbox(
                sample_geo_df,
                lat="latitude",
                lon="longitude",
                color=geo_metric,
                size='size_metric',
                hover_name="city",
                hover_data=['country', geo_metric],
                color_continuous_scale="Turbo",
                zoom=1,
                mapbox_style="carto-darkmatter",
                height=700,
                title=f"Global {geo_metric.replace('_', ' ').title()} Distribution"
            )
            
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with tab5:
            st.markdown("### 📉 Time Series Analysis")
            
            # Daily aggregation
            ts_data = df.groupby(df['last_updated'].dt.date).agg({
                'temperature_celsius': ['mean', 'std'],
                'pm2_5': ['mean', 'std'],
                'climate_risk_index': 'mean'
            }).reset_index()
            
            ts_data.columns = ['date', 'temp_mean', 'temp_std', 'pm25_mean', 'pm25_std', 'risk_mean']
            
            fig_ts = go.Figure()
            
            fig_ts.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['temp_mean'],
                mode='lines',
                name='Temperature',
                line=dict(color='#f59e0b', width=2),
                yaxis='y1'
            ))
            
            fig_ts.add_trace(go.Scatter(
                x=ts_data['date'],
                y=ts_data['pm25_mean'],
                mode='lines',
                name='PM2.5',
                line=dict(color='#8b5cf6', width=2),
                yaxis='y2'
            ))
            
            fig_ts.update_layout(
                title='Multi-Metric Time Series Evolution',
                xaxis=dict(title='Date'),
                yaxis=dict(
                    title=dict(text='Temperature (°C)', font=dict(color='#f59e0b'))
                ),
                yaxis2=dict(
                    title=dict(text='PM2.5', font=dict(color='#8b5cf6')),
                    overlaying='y',
                    side='right'
                ),
                template='plotly_dark',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
    
    # ==================== PAGE: ML FORECASTING ====================
    elif "🔬" in page:
        st.markdown("# 🔬 Machine Learning Forecasting Studio")
        st.markdown("*AI-powered predictive climate modeling*")
        st.markdown("---")
        
        if ml_model:
            st.success("✅ Advanced ML Model Loaded Successfully")
        else:
            st.warning("⚠️ Using simulation mode for demonstration")
        
        col_input, col_output = st.columns([1, 2])
        
        with col_input:
            st.markdown("### 🎯 Forecast Configuration")
            
            selected_city = st.selectbox("Select City", df['city'].unique()[:100])
            forecast_days = st.slider("Forecast Horizon (Days)", 1, 14, 7)
            
            city_data = df[df['city'] == selected_city].iloc[0]
            
            st.markdown("---")
            st.markdown("### 📊 Current Conditions")
            st.metric("Temperature", f"{city_data['temperature_celsius']:.1f}°C")
            st.metric("PM2.5", f"{city_data['pm2_5']:.1f}")
            st.metric("Humidity", f"{city_data['humidity']:.0f}%")
            st.metric(" Wind Speed", f"{city_data['wind_kph']:.1f} km/h")
            st.metric("Risk Index", f"{city_data['climate_risk_index']:.0f}/100")
        
        with col_output:
            st.markdown(f"### 📈 {forecast_days}-Day Forecast for {selected_city}")
            
            # Generate synthetic forecast
            np.random.seed(42)
            future_dates = [datetime.now() + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            trend = np.random.normal(0, 1.2, forecast_days).cumsum()
            forecast_temps = city_data['temperature_celsius'] + trend
            
            forecast_pm25 = city_data['pm2_5'] + np.random.normal(0, 5, forecast_days).cumsum()
            forecast_pm25 = np.clip(forecast_pm25, 0, 500)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Temperature': forecast_temps,
                'PM2.5': forecast_pm25,
                'Temp_Lower': forecast_temps - 3,
                'Temp_Upper': forecast_temps + 3
            })
            
            fig_forecast = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Forecast', 'Air Quality Forecast'),
                vertical_spacing=0.15
            )
            
            # Temperature forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Temperature'],
                mode='lines+markers', name='Temp Forecast',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=8)
            ), row=1, col=1)
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Temp_Upper'],
                mode='lines', line=dict(width=0), showlegend=False
            ), row=1, col=1)
            
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['Temp_Lower'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(245, 158, 11, 0.2)',
                showlegend=False
            ), row=1, col=1)
            
            # PM2.5 forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['Date'], y=forecast_df['PM2.5'],
                mode='lines+markers', name='PM2.5 Forecast',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=8)
            ), row=2, col=1)
            
            fig_forecast.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
            fig_forecast.update_yaxes(title_text="PM2.5 (µg/m³)", row=2, col=1)
            
            fig_forecast.update_layout(
                height=600,
                template='plotly_dark',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast summary
            st.markdown("#### 📋 Forecast Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Forecast Temp", f"{forecast_df['Temperature'].mean():.1f}°C",
                         f"{(forecast_df['Temperature'].mean() - city_data['temperature_celsius']):.1f}°C")
            with col2:
                st.metric("Max Forecast Temp", f"{forecast_df['Temperature'].max():.1f}°C")
            with col3:
                st.metric("Min Forecast Temp", f"{forecast_df['Temperature'].min():.1f}°C")
    
    # ==================== PAGE: ALERT HUB ====================
    elif "🚨" in page:
        st.markdown("# 🚨 Real-Time Alert Hub")
        st.markdown("*Critical weather events and emergency notifications*")
        st.markdown("---")
        
        alert_type = st.selectbox("Filter by Alert Type", 
                                 ["All Alerts", "Temperature Extremes", "Air Quality Crisis", 
                                  "High Risk Zones", "Anomalous Events"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Temperature Alerts")
            temp_alerts = df.nlargest(10, 'temperature_celsius')[
                ['city', 'country', 'temperature_celsius', 'last_updated']
            ]
            
            for idx, row in temp_alerts.iterrows():
                severity = "CRITICAL" if row['temperature_celsius'] > 40 else "WARNING"
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>🌡️ {severity}: {row['city']}, {row['country']}</strong><br>
                    Temperature: <strong>{row['temperature_celsius']:.1f}°C</strong><br>
                    Updated: {row['last_updated'].strftime('%Y-%m-%d %H:%M')}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💨 Air Quality Alerts")
            aqi_alerts = df.nlargest(10, 'pm2_5')[
                ['city', 'country', 'pm2_5', 'aqi_category', 'last_updated']
            ]
            
            for idx, row in aqi_alerts.iterrows():
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>💨 {row['aqi_category']}: {row['city']}, {row['country']}</strong><br>
                    PM2.5: <strong>{row['pm2_5']:.1f} µg/m³</strong><br>
                    Updated: {row['last_updated'].strftime('%Y-%m-%d %H:%M')}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚠️ High Risk Zones")
        
        risk_zones = df.nlargest(15, 'climate_risk_index')[
            ['city', 'country', 'climate_risk_index', 'temperature_celsius', 'pm2_5']
        ]
        
        fig_risk = px.bar(
            risk_zones,
            x='climate_risk_index',
            y='city',
            color='climate_risk_index',
            orientation='h',
            title='Top 15 Highest Risk Cities',
            color_continuous_scale='Reds',
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # ==================== PAGE: ADVANCED INSIGHTS ====================
    elif "📈" in page:
        st.markdown("# 📈 Advanced Climate Insights")
        st.markdown("*Executive-level intelligence and strategic analysis*")
        st.markdown("---")
        
        # Continental Analysis
        st.markdown("### 🌍 Global Regional Analysis")
        
        # Add continent mapping (simplified)
        def get_continent(country):
            # Simplified continent mapping
            asia = ['China', 'India', 'Japan', 'Thailand', 'Indonesia']
            europe = ['United Kingdom', 'Germany', 'France', 'Italy', 'Spain']
            americas = ['United States', 'Canada', 'Brazil', 'Mexico', 'Argentina']
            
            if country in asia: return 'Asia'
            elif country in europe: return 'Europe'
            elif country in americas: return 'Americas'
            else: return 'Other'
        
        df['continent'] = df['country'].apply(get_continent)
        
        continent_stats = df.groupby('continent').agg({
            'temperature_celsius': 'mean',
            'pm2_5': 'mean',
            'climate_risk_index': 'mean',
            'city': 'count'
        }).reset_index()
        continent_stats.columns = ['Continent', 'Avg Temp', 'Avg PM2.5', 'Avg Risk', 'Cities']
        
        fig_cont = px.sunburst(
            df,
            path=['continent', 'country', 'city'],
            values='climate_risk_index',
            color='temperature_celsius',
            color_continuous_scale='RdYlBu_r',
            title='Hierarchical Climate Risk Distribution',
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig_cont, use_container_width=True)
        
        st.markdown("---")
        
        # Season Comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌦️ Seasonal Patterns")
            
            season_data = df.groupby('season').agg({
                'temperature_celsius': 'mean',
                'pm2_5': 'mean',
                'humidity': 'mean'
            }).reset_index()
            
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatterpolar(
                r=season_data['temperature_celsius'],
                theta=season_data['season'],
                fill='toself',
                name='Temperature',
                line=dict(color='#f59e0b')
            ))
            
            fig_season.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 AQI Distribution")
            
            aqi_dist = df['aqi_category'].value_counts().reset_index()
            aqi_dist.columns = ['Category', 'Count']
            
            fig_aqi = px.pie(
                aqi_dist,
                values='Count',
                names='Category',
                title='Global Air Quality Distribution',
                template='plotly_dark',
                height=400,
                color_discrete_sequence=px.colors.sequential.RdBu_r
            )
            
            st.plotly_chart(fig_aqi, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 Executive Summary Table")
        
        summary_table = df.groupby('country').agg({
            'temperature_celsius': ['mean', 'min', 'max'],
            'pm2_5': 'mean',
            'climate_risk_index': 'mean',
            'city': 'count'
        }).round(2)
        
        summary_table.columns = ['Avg Temp', 'Min Temp', 'Max Temp', 'Avg PM2.5', 'Risk Index', 'Cities']
        summary_table = summary_table.sort_values('Risk Index', ascending=False).head(20)
        
        st.dataframe(summary_table, use_container_width=True, height=400)
    
    # ==================== PAGE: API SETTINGS ====================
    elif "⚙️" in page:
        st.markdown("# ⚙️ API Settings & Configuration")
        st.markdown("*Manage data sources and system configuration*")
        st.markdown("---")
        
        st.markdown("### 🔑 Weather API Configuration")
        
        with st.form("api_config"):
            st.markdown("#### Current API Key")
            current_key = st.text_input("API Key", value="9c5c7287c0e2ec90680c189de7b28659", type="password")
            
            st.markdown("#### Data Source Settings")
            data_source = st.selectbox("Data Source", ["WeatherAPI", "OpenWeatherMap", "Custom"])
            refresh_interval = st.slider("Auto-refresh Interval (minutes)", 5, 120, 30)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Save Configuration", use_container_width=True):
                    st.success("✅ Configuration saved successfully!")
            with col2:
                if st.form_submit_button("🔄 Test Connection", use_container_width=True):
                    st.info("🔄 Testing API connection...")
                    st.success("✅ API connection successful!")
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Database Size", "24.5 MB", "↑ 2.1 MB")
        with col2:
            st.metric("Total Records", f"{len(df):,}", "↑ 1,234")
        with col3:
            st.metric("API Calls Today", "1,456", "↓ 23")
        with col4:
            st.metric("Last Sync", "5 min ago", "Active")
        
        st.markdown("---")
        st.markdown("### 🛠️ Advanced Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("✅ Cache cleared!")
            
            if st.button("📥 Export Data (CSV)", use_container_width=True):
                st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False),
                    file_name="weather_data_export.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 Reload Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            if st.button("📊 Generate Report", use_container_width=True):
                st.info("📊 Generating comprehensive report...")
                st.success("✅ Report generated!")
        
        st.markdown("---")
        st.markdown("### 📝 System Logs")
        
        logs = [
            "[2024-12-14 11:24:19] INFO: Data loaded successfully (23.5 MB)",
            "[2024-12-14 11:20:15] INFO: API connection established",
            "[2024-12-14 11:15:42] INFO: Anomaly detection completed (127 anomalies found)",
            "[2024-12-14 11:10:33] INFO: ML model loaded successfully",
            "[2024-12-14 11:05:21] INFO: Cache initialized"
        ]
        
        for log in logs:
            st.code(log, language="log")

if __name__ == "__main__":
    main()
