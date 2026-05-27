import streamlit as st
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyIQ Nigeria",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background: #0a0f0d;
    color: #e8f0ea;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111a14 !important;
    border-right: 1px solid #1e2e21;
}
[data-testid="stSidebar"] * {
    color: #e8f0ea !important;
}

/* Inputs */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div > div {
    background: #162019 !important;
    border: 1px solid #1e2e21 !important;
    border-radius: 8px !important;
    color: #e8f0ea !important;
}

/* Button */
.stButton > button {
    background: #4cff8f !important;
    color: #0a0f0d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #6bffaa !important;
    transform: translateY(-1px) !important;
}

/* Metric cards */
.metric-card {
    background: #111a14;
    border: 1px solid #1e2e21;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 12px;
    color: #6b8a70;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-family: 'Syne', sans-serif;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1px;
}
.metric-sub {
    font-size: 13px;
    color: #6b8a70;
    margin-top: 4px;
}
.green { color: #4cff8f; }
.orange { color: #ff6b35; }
.yellow { color: #ffd166; }
.white { color: #e8f0ea; }

/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.05;
    color: #e8f0ea;
}
.hero-title span { color: #4cff8f; }
.hero-sub {
    font-size: 16px;
    color: #6b8a70;
    margin-top: 8px;
    line-height: 1.6;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(76,255,143,0.08);
    border: 1px solid rgba(76,255,143,0.2);
    color: #4cff8f;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 16px;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.5px;
}

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #6b8a70;
    margin-bottom: 16px;
    margin-top: 24px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2e21;
}

/* Insight box */
.insight-box {
    background: rgba(76,255,143,0.05);
    border: 1px solid rgba(76,255,143,0.15);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 8px;
    font-size: 14px;
    color: #b8d4bc;
    line-height: 1.6;
}
.insight-box strong { color: #4cff8f; }

/* Warning box */
.warning-box {
    background: rgba(255,107,53,0.06);
    border: 1px solid rgba(255,107,53,0.2);
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 8px;
    font-size: 14px;
    color: #f0c0a8;
    line-height: 1.6;
}
.warning-box strong { color: #ff6b35; }

/* Divider */
.divider {
    height: 1px;
    background: #1e2e21;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

# ─── Constants ─────────────────────────────────────────────────────────────────
BUILDING_TYPES = {
    'Education': 0,
    'Entertainment/public assembly': 1,
    'Food sales and service': 2,
    'Healthcare': 3,
    'Lodging/residential': 4,
    'Manufacturing/industrial': 5,
    'Office': 6,
    'Other': 7,
    'Parking': 8,
    'Public services': 9,
    'Religious worship': 10,
    'Retail': 11,
    'Services': 12,
    'Technology/science': 13,
    'Utility': 14,
    'Warehouse/storage': 15
}

OPERATING_HOURS = {
    'Education': 9, 'Entertainment/public assembly': 10,
    'Food sales and service': 12, 'Healthcare': 24,
    'Lodging/residential': 24, 'Manufacturing/industrial': 16,
    'Office': 8, 'Other': 8, 'Parking': 18,
    'Public services': 8, 'Religious worship': 6,
    'Retail': 12, 'Services': 8, 'Technology/science': 10,
    'Utility': 24, 'Warehouse/storage': 10
}

ZONES = {
    'South West': {
        'states': 'Lagos, Ogun, Oyo, Osun, Ondo, Ekiti',
        'temp': 30.5, 'wind': 3.2, 'pressure': 1010.5,
        'grid_min': 8, 'grid_max': 10,
        'hint': '~8–10 hrs/day',
        'overlap': 0.65
    },
    'South South': {
        'states': 'Rivers, Delta, Edo, Cross River, Akwa Ibom, Bayelsa',
        'temp': 31.0, 'wind': 2.8, 'pressure': 1009.0,
        'grid_min': 3, 'grid_max': 6,
        'hint': '~3–6 hrs/day',
        'overlap': 0.55
    },
    'South East': {
        'states': 'Enugu, Anambra, Imo, Abia, Ebonyi',
        'temp': 29.5, 'wind': 2.5, 'pressure': 1010.0,
        'grid_min': 9, 'grid_max': 10,
        'hint': '~9–10 hrs/day',
        'overlap': 0.70
    },
    'North West': {
        'states': 'Kano, Kaduna, Katsina, Zamfara, Kebbi, Sokoto, Jigawa',
        'temp': 33.0, 'wind': 4.5, 'pressure': 950.0,
        'grid_min': 7, 'grid_max': 8,
        'hint': '~7–8 hrs/day',
        'overlap': 0.60
    },
    'North East': {
        'states': 'Borno, Yobe, Adamawa, Gombe, Bauchi, Taraba',
        'temp': 34.0, 'wind': 5.0, 'pressure': 945.0,
        'grid_min': 3, 'grid_max': 5,
        'hint': '~3–5 hrs/day',
        'overlap': 0.50
    },
    'North Central': {
        'states': 'FCT Abuja, Niger, Kwara, Kogi, Benue, Plateau, Nasarawa',
        'temp': 29.0, 'wind': 2.8, 'pressure': 912.0,
        'grid_min': 7, 'grid_max': 11,
        'hint': '~7–11 hrs/day',
        'overlap': 0.75
    }
}

DIESEL_PRICE_PER_LITER = 1200.0

# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:28px">Energy<span>IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:#6b8a70;margin-bottom:24px;text-transform:uppercase;letter-spacing:0.5px">Nigeria Energy Predictor</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Building Details</div>', unsafe_allow_html=True)

    building_type = st.selectbox("Building Type", list(BUILDING_TYPES.keys()), index=6)
    square_feet = st.number_input("Floor Area (sq ft)", min_value=100, max_value=500000, value=5000, step=500)
    floor_count = st.number_input("Number of Floors", min_value=1, max_value=50, value=3)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2024, value=2005)
    building_age = 2024 - year_built

    st.markdown('<div class="section-label">Time</div>', unsafe_allow_html=True)
    month = st.slider("Month", 1, 12, 7, format="%d")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    st.caption(f"Selected: {month_names[month-1]}")
    day_type = st.radio("Day Type", ["Weekday", "Weekend"], horizontal=True)
    is_weekend = 1 if day_type == "Weekend" else 0

    st.markdown('<div class="section-label">Nigerian Context</div>', unsafe_allow_html=True)
    zone = st.selectbox("Geopolitical Zone", list(ZONES.keys()))
    zone_info = ZONES[zone]
    st.caption(f"States: {zone_info['states']}")
    st.caption(f"Typical grid supply: {zone_info['hint']}")

    default_grid = (zone_info['grid_min'] + zone_info['grid_max']) / 2
    grid_hours = st.slider("Your Grid Hours Per Day", 0.0, 24.0, float(default_grid), step=0.5)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Energy Consumption", type="primary")

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="badge">XGBoost Model · R² = 0.87 · Trained on 500k ASHRAE Records</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Predict <span>energy</span><br>consumption.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter building details to predict daily electricity demand and understand your grid-diesel energy balance.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Model could not be loaded. Make sure best_model.pkl is in the same folder as this file. Error: {model_error}")
    st.stop()

# ─── Prediction Logic ──────────────────────────────────────────────────────────
if predict_btn:
    climate = ZONES[zone]
    air_temperature = climate['temp']
    wind_speed = climate['wind']
    sea_level_pressure = climate['pressure']

    hour = 12
    business_hours = 1 if is_weekend == 0 else 0
    nigerian_dry_season = 1 if month in [11, 12, 1, 2, 3] else 0
    primary_use_encoded = BUILDING_TYPES.get(building_type, 7)

    features = np.array([[
        hour, month, is_weekend, business_hours, nigerian_dry_season,
        square_feet, floor_count, building_age, primary_use_encoded,
        air_temperature, wind_speed, sea_level_pressure
    ]])

    log_pred = model.predict(features)[0]
    total_demand_kwh = float(np.expm1(log_pred))

# ─── Weekday prediction for monthly calculation ──────────────────────────────────────────────────────────
    features_weekday = np.array([[
        hour, month, 0, 1, nigerian_dry_season,
        square_feet, floor_count, building_age, primary_use_encoded,
        air_temperature, wind_speed, sea_level_pressure
    ]])
    demand_weekday = float(np.expm1(model.predict(features_weekday)[0]))

# ─── Weekend prediction for monthly calculation ──────────────────────────────────────────────────────────
    features_weekend = np.array([[
        hour, month, 1, 0, nigerian_dry_season,
        square_feet, floor_count, building_age, primary_use_encoded,
        air_temperature, wind_speed, sea_level_pressure
    ]])
    demand_weekend = float(np.expm1(model.predict(features_weekend)[0]))

    # ─── Generator hours ──────────────────────────────────────────────────────────
    operating_hrs = OPERATING_HOURS.get(building_type, 8)

    # ─── Grid supply (adjusted for overlap efficiency) ──────────────────────────────────────────────────────────
    overlap_efficiency = ZONES[zone]['overlap']
    effective_grid_hours = grid_hours * overlap_efficiency
    grid_supply_kwh = total_demand_kwh * (effective_grid_hours / 24)
    diesel_demand_kwh = max(0, total_demand_kwh - grid_supply_kwh)
    grid_coverage_pct = (grid_supply_kwh / total_demand_kwh * 100) if total_demand_kwh > 0 else 0

    # ─── Deficit hours ──────────────────────────────────────────────────────────
    deficit = max(0, 24 - effective_grid_hours)
    generator_hours = round(min(24, deficit + 1.0), 1)

    # ─── Diesel cost ──────────────────────────────────────────────────────────
    litres_per_hour = max(0.5, square_feet * 0.0015)
    diesel_litres = round(generator_hours * litres_per_hour, 2)
    diesel_cost_ngn = round(diesel_litres * DIESEL_PRICE_PER_LITER, 2)

    # ─── Monthly ──────────────────────────────────────────────────────────
    weekday_kwh = total_demand_kwh
    weekend_kwh = total_demand_kwh * 0.4
    monthly_kwh = round((demand_weekday * 22) + (demand_weekend * 8), 2)
    monthly_diesel_cost = round(diesel_cost_ngn * 22, 2)
       
    season = "Dry Season (Harmattan)" if nigerian_dry_season else "Rainy Season"

    # ─── Results Layout ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Energy Demand</div>
            <div class="metric-value green">{total_demand_kwh:,.2f}</div>
            <div class="metric-sub">kWh / day (full capacity)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Grid Coverage</div>
            <div class="metric-value white">{grid_supply_kwh:,.2f}</div>
            <div class="metric-sub">kWh / day · {grid_coverage_pct:.0f}% of demand</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Diesel Energy Gap</div>
            <div class="metric-value orange">{diesel_demand_kwh:,.2f}</div>
            <div class="metric-sub">kWh / day must come from generator</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("What does this mean?")

    if total_demand_kwh < 30:
        st.info("💡 **Low Consumption:** Equivalent to a small office or shop. Primarily lighting, fans, and basic IT equipment.")
    elif total_demand_kwh < 100:
        st.warning("🏢 **Moderate Consumption:** Typical for a medium office. Includes essential cooling (AC) and office infrastructure.")
    else:
        st.error("⚡ **High Consumption:** Suggests heavy industrial use, large-scale HVAC, or significant machinery.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


    # ─── Donut Chart ─────────────────────────────────────────────────────────
    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        fig2 = go.Figure(go.Pie(
            values=[grid_supply_kwh, diesel_demand_kwh],
            labels=['Grid Supply', 'Diesel Requirement'],
            hole=0.65,
            marker_colors=["#4cff8ec9", "#ef7144"],
            textinfo='percent',
            textfont=dict(size=13, family='Syne'),
        ))
        fig2.update_layout(
            paper_bgcolor='#0a0f0d',
            font=dict(color='#e8f0ea', family='DM Sans'),
            legend=dict(bgcolor='#111a14', bordercolor='#1e2e21', borderwidth=1),
            height=260,
            margin=dict(l=0, r=0, t=10, b=10),
            annotations=[dict(
                text=f'{grid_coverage_pct:.0f}%<br>Grid',
                x=0.5, y=0.5, font_size=18,
                font=dict(color='#4cff8f', family='Syne', size=20),
                showarrow=False
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_d2:
        st.markdown('<div class="section-label">Nigerian Grid Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:8px">
            <div class="metric-label">Zone</div>
            <div style="font-family:Syne;font-size:16px;font-weight:700;color:#4cff8f">{zone}</div>
            <div class="metric-sub">{zone_info['states']}</div>
        </div>
        <div class="metric-card" style="margin-bottom:8px">
            <div class="metric-label">Season</div>
            <div style="font-family:Syne;font-size:16px;font-weight:700;color:#e8f0ea">{season}</div>
        </div>
        <div class="metric-card" style="margin-bottom:8px">
            <div class="metric-label">Generator Hours Needed</div>
            <div style="font-family:Syne;font-size:22px;font-weight:800;color:#ffd166">{generator_hours:.1f} hrs</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Est. Daily Diesel Cost</div>
            <div style="font-family:Syne;font-size:22px;font-weight:800;color:#ff6b35">₦{diesel_cost_ngn:,.0f}</div>
            <div class="metric-sub">{diesel_litres:.1f} litres @ ₦1,200/L</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ─── Monthly Projection ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Monthly Projection</div>', unsafe_allow_html=True)
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Energy Demand</div>
            <div class="metric-value green">{monthly_kwh:,.0f}</div>
            <div class="metric-sub">kWh / month</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monthly Diesel Cost</div>
            <div class="metric-value orange">₦{monthly_diesel_cost:,.0f}</div>
            <div class="metric-sub">estimated generator spend</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        annual_diesel = monthly_diesel_cost * 12
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Diesel Projection</div>
            <div class="metric-value yellow">₦{annual_diesel:,.0f}</div>
            <div class="metric-sub">estimated yearly spend</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Insight Box ─────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if grid_coverage_pct < 40:
        st.markdown(f"""
        <div class="warning-box">
            <strong>High Diesel Dependency Alert:</strong> This {building_type.lower()} in the {zone} zone 
            is meeting only <strong>{grid_coverage_pct:.0f}%</strong> of its energy demand from the grid. 
            With {grid_hours} hours of grid supply per day, the building relies heavily on diesel generation. 
            Annual diesel expenditure is projected at <strong>₦{annual_diesel:,.0f}</strong>. 
            Consider energy efficiency upgrades or solar backup systems to reduce generator dependency.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Energy Profile:</strong> This {building_type.lower()} in the {zone} zone 
            has a total daily demand of <strong>{total_demand_kwh:,.2f} kWh</strong>. 
            With {grid_hours} hours of grid supply, <strong>{grid_coverage_pct:.0f}%</strong> of demand 
            is met by the grid. The remaining <strong>{100-grid_coverage_pct:.0f}%</strong> requires 
            diesel generation, costing approximately <strong>₦{monthly_diesel_cost:,.0f}</strong> per month.
        </div>
        """, unsafe_allow_html=True)

else:
    # ─── Empty State ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;color:#6b8a70;">
        <div style="font-size:64px;margin-bottom:16px;opacity:0.4">⚡</div>
        <div style="font-family:Syne;font-size:20px;font-weight:700;margin-bottom:8px;color:#e8f0ea">Ready to predict</div>
        <div style="font-size:15px;line-height:1.6">Fill in the building details in the sidebar<br>and click <strong style="color:#4cff8f">Predict Energy Consumption</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Model info at bottom
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Information</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    with col_i1:
        st.markdown('<div class="metric-card"><div class="metric-label">Algorithm</div><div style="font-family:Syne;font-weight:700;color:#4cff8f">XGBoost</div></div>', unsafe_allow_html=True)
    with col_i2:
        st.markdown('<div class="metric-card"><div class="metric-label">R² Score</div><div style="font-family:Syne;font-weight:700;color:#4cff8f">0.8749</div></div>', unsafe_allow_html=True)
    with col_i3:
        st.markdown('<div class="metric-card"><div class="metric-label">Training Data</div><div style="font-family:Syne;font-weight:700;color:#e8f0ea">500k Records</div></div>', unsafe_allow_html=True)
    with col_i4:
        st.markdown('<div class="metric-card"><div class="metric-label">Climate Source</div><div style="font-family:Syne;font-weight:700;color:#e8f0ea">NASA POWER</div></div>', unsafe_allow_html=True)
