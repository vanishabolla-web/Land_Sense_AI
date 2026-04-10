# ===============================
# Imports
# ===============================
import os
import numpy as np
import pandas as pd
import joblib
import gdown
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Land Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ===================== Light Background ===================== */
.stApp {
    background: linear-gradient(135deg, #f7f9fc 0%, #eef2f7 100%);
}

/* ===================== Sidebar ===================== */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid rgba(255, 183, 3, 0.4);
}

/* ===================== Titles ===================== */
h1 { 
    color: #fb8500 !important; 
    letter-spacing: 1px;
}

h2, h3 {
    color: #ffb703 !important;
}

/* ===================== Section Badge ===================== */
.section-badge {
    background: rgba(255, 183, 3, 0.15);
    border: 1px solid rgba(255, 183, 3, 0.5);
    color: #fb8500;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
    display: inline-block;
}

/* ===================== Metric Cards ===================== */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 2px solid rgba(255, 183, 3, 0.35);
    border-radius: 14px;
    padding: 18px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

[data-testid="metric-container"] label {
    color: #fb8500 !important;
    font-weight: 600;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #333 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 24px !important;
}

/* ===================== Buttons ===================== */
.stButton > button {
    background: linear-gradient(135deg, #ffb703, #fb8500);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 14px 40px;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 10px rgba(255,183,3,0.5);
}

/* ===================== Info Cards ===================== */
.info-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    color: #444;
    padding: 16px;
    border-radius: 12px;
}

/* ===================== Inputs ===================== */
input, textarea, select {
    background-color: #ffffff !important;
    color: #333 !important;
    border: 1px solid rgba(255,183,3,0.4) !important;
}

/* ===================== Radio / Selectbox text ===================== */
div[data-baseweb="select"] > div {
    color: #333 !important;
}

/* ===================== Horizontal line ===================== */
hr {
    border: 1px solid rgba(255,183,3,0.25);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Data & Pipeline
# ===============================
inpdata = pd.read_csv('final_land_price_65k.csv')

@st.cache_resource
def load_model():
    return joblib.load("landprice.pkl")

pipeline = load_model()

# Extract components
model = pipeline["model"]
encoder = pipeline["encoder"]
scaler_X = pipeline["scaler_X"]
scaler_y = pipeline["scaler_y"]
columns_order = pipeline["columns"]
categorical_cols = pipeline["categorical_cols"]

# ===============================
# City Coordinates for Map
# ===============================
CITY_COORDS = {
    "Jaipur": (26.9124, 75.7873), "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025), "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867), "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567), "Ahmedabad": (23.0225, 72.5714),
    "Kolkata": (22.5726, 88.3639), "Surat": (21.1702, 72.8311),
    "Lucknow": (26.8467, 80.9462), "Nagpur": (21.1458, 79.0882),
    "Indore": (22.7196, 75.8577), "Bhopal": (23.2599, 77.4126),
    "Visakhapatnam": (17.6868, 83.2185), "Chandigarh": (30.7333, 76.7794),
    "Kochi": (9.9312, 76.2673), "Coimbatore": (11.0168, 76.9558),
}

def get_city_coords(city_name):
    for key, coords in CITY_COORDS.items():
        if key.lower() in city_name.lower() or city_name.lower() in key.lower():
            return coords
    return (20.5937, 78.9629)  # Center of India fallback

# ===============================
# Helper: Format INR
# ===============================
def format_inr(amount):
    """Format amount in Indian numbering system with Cr/Lac suffix."""
    if amount >= 1e7:
        return f"₹{amount/1e7:.2f} Cr"
    elif amount >= 1e5:
        return f"₹{amount/1e5:.2f} Lac"
    else:
        return f"₹{amount:,.0f}"

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.markdown("""
<div style='padding: 10px 0 20px 0;'>
    <div style='color:#00e5a0; font-size:1.3rem; font-weight:700; letter-spacing:-0.5px;'>🌿 LandSense</div>
    <div style='color:#4a7a68; font-size:0.75rem; margin-top:4px;'>AI-Powered Valuation</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Go to", ["📄 Description", "🏡 Prediction"])

st.sidebar.markdown("""
<div style='color:#4a7a68; font-size:0.80rem; line-height:1.6;'>

<b style='color:#7ecfb3;'>About</b><br>
Predicts future land prices using ML trained on 65,000+ records across India.
<br><br>

<b style='color:#7ecfb3;'>Data Coverage</b><br>
Multiple states · All zoning types · 2024–2026 projection
<br><br>

<b style='color:#ff6b6b;'>Disclaimer</b><br>
This model provides estimated predictions based on historical data and machine learning algorithms. 
The results may not be fully accurate and should not be considered as financial or investment advice. 
Always verify with real-world data and experts before making decisions.

</div>
""", unsafe_allow_html=True)

# ===============================
# 📄 Description Page
# ===============================
if page == "📄 Description":
    st.markdown("# 🌿 LandSense — AI Land Valuation")
    st.markdown("---")
    st.markdown("""
    <div class='info-card'>
    Welcome to <b style='color:#00e5a0;'>LandSense</b> — an AI-powered platform for predicting future land prices 
    across India. Enter property details, and our model provides a projected price for 2026, total land valuation, 
    ROI analysis, and investment-grade recommendations for buyers and real estate agents.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Records", "65,000+")
    col2.metric("States Covered", str(inpdata['state'].nunique()))
    col3.metric("Zoning Types", str(inpdata['zoning'].nunique()))

# ===============================
# 🏡 Prediction Page
# ===============================
elif page == "🏡 Prediction":
    st.markdown("# Note: If you are a Buyer give your assumptions, if Agent use your actual values")
    st.markdown("# 🏡 Future Land Price Prediction")
    st.markdown("---")

    # ── Location Section ──────────────────────────────
    st.markdown('<div class="section-badge">📍 Location</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", sorted(inpdata['state'].unique()))
    with col2:
        cities = inpdata[inpdata['state'] == state]['city'].unique()
        city = st.selectbox("City", sorted(cities))
    with col3:
        localities = inpdata[
            (inpdata['state'] == state) & (inpdata['city'] == city)
        ]['locality'].unique()
        locality = st.selectbox("Locality", sorted(localities))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Property Details ──────────────────────────────
    st.markdown('<div class="section-badge">🏗️ Property Details</div>', unsafe_allow_html=True)
    col4, col5 = st.columns(2)
    zoning = col4.selectbox("Zoning", inpdata['zoning'].unique())
    city_tier = col5.number_input("City Tier (1–3)", 1, 3)

    col6, col7 = st.columns(2)
    land_area = col6.number_input("Land Area (sqft)", min_value=0.0, step=100.0)
    dist_city = col7.number_input("Distance from City Center (km)", min_value=0.0, step=0.1)

    col8, col9 = st.columns(2)
    dist_highway = col8.number_input("Distance from Highway (km)", min_value=0.0, step=0.1)
    dist_transport = col9.number_input("Distance from Transport (km)", min_value=0.0, step=0.1)

    col10, col11 = st.columns(2)
    dist_amenities = col10.number_input("Distance to Amenities (km)", min_value=0.0, step=0.1)
    historical_growth = col11.number_input("Historical Growth (%)", step=0.1)

    col12, col13 = st.columns(2)
    population_growth = col12.number_input("Population Growth (%)", step=0.1)
    road_quality = col13.number_input("Road Quality Score", step=0.1)

    col14, col15 = st.columns(2)
    utility_access = col14.selectbox("Utility Access", [0, 1], format_func=lambda x: "0" if x else "1")
    govt_dev_plan = col15.selectbox("Government Development Plan", [0, 1], format_func=lambda x: "0" if x else "1")

    col16, col17 = st.columns(2)
    flood_risk = col16.selectbox("Flood Risk", [0, 1], format_func=lambda x: "0" if x else "1")
    current_price = col17.number_input("Current Market Price per Sqft (₹)", min_value=0.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────
    predict_clicked = st.button("🔮 Predict Future Price")

    if predict_clicked:

        row = pd.DataFrame([[
            state, city, locality, city_tier, zoning, land_area,
            dist_city, dist_highway, dist_transport, dist_amenities,
            historical_growth, population_growth, road_quality,
            utility_access, govt_dev_plan, flood_risk
        ]], columns=columns_order)

        try:
            row[categorical_cols] = encoder.transform(row[categorical_cols])
            row_scaled = scaler_X.transform(row)
            y_scaled = model.predict(row_scaled)
            predicted_price = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[0][0]

            diff = predicted_price - current_price
            roi = (diff / current_price) * 100 if current_price else 0

            # Total land prices
            total_current  = land_area * current_price
            total_predicted = land_area * predicted_price
            total_diff     = total_predicted - total_current

            st.markdown("---")
             
            # ── 2. PRICE PER SQFT ANALYSIS ───────────────────
            st.markdown('<div class="section-badge">📊 Price per Sqft Analysis</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Price/sqft", f"₹{current_price:,.2f}")
            m2.metric("Predicted Price/sqft", f"₹{predicted_price:,.2f}", delta=f"+₹{diff:,.2f}")
            m3.metric("Difference/sqft", f"₹{diff:,.2f}")
            m4.metric("ROI %", f"{roi:,.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 1. TOTAL LAND VALUE ───────────────────────────
            st.markdown('<div class="section-badge">💎 Total Land Valuation</div>', unsafe_allow_html=True)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown(f"""
                <div class='total-price-box'>
                    <div class='label'>Current Total Value</div>
                    <div class='amount'>{format_inr(total_current)}</div>
                    <div class='sub'>₹{current_price:,.2f}/sqft × {land_area:,.0f} sqft</div>
                </div>
                """, unsafe_allow_html=True)
            with tc2:
                st.markdown(f"""
                <div class='total-price-box' style='border-color:rgba(0,229,160,0.6); background:linear-gradient(135deg,rgba(0,229,160,0.18),rgba(0,180,120,0.10));'>
                    <div class='label'>Predicted Total (2026)</div>
                    <div class='amount'>{format_inr(total_predicted)}</div>
                    <div class='sub'>₹{predicted_price:,.2f}/sqft × {land_area:,.0f} sqft</div>
                </div>
                """, unsafe_allow_html=True)
            with tc3:
                gain_color = "#00e5a0" if total_diff >= 0 else "#ff6b6b"
                st.markdown(f"""
                <div class='total-price-box' style='border-color:rgba(0,229,160,0.3);'>
                    <div class='label'>Projected Gain</div>
                    <div class='amount' style='color:{gain_color};'>{format_inr(abs(total_diff))}</div>
                    <div class='sub'>{"▲ Appreciation" if total_diff >= 0 else "▼ Depreciation"} on full land</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 3. CHARTS ROW ─────────────────────────────────
            chart_col1, chart_col2 = st.columns([1, 1])

            # ROI Gauge
            with chart_col1:
                st.markdown('<div class="section-badge">🎯 ROI Gauge</div>', unsafe_allow_html=True)
                gauge_color = (
                    "#00e5a0" if roi > 75 else
                    "#f0c040" if roi > 25 else
                    "#ff6b6b"
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(roi, 2),
                    number={'suffix': "%", 'font': {'size': 36, 'color': '#ffffff', 'family': 'DM Mono'}},
                    delta={'reference': 50, 'increasing': {'color': '#00e5a0'}, 'decreasing': {'color': '#ff6b6b'}},
                    title={'text': "Return on Investment", 'font': {'size': 14, 'color': '#7ecfb3'}},
                    gauge={
                        'axis': {'range': [0, 150], 'tickcolor': '#4a7a68', 'tickfont': {'color': '#7ecfb3', 'size': 10}},
                        'bar': {'color': gauge_color, 'thickness': 0.25},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 25],   'color': 'rgba(255,107,107,0.15)'},
                            {'range': [25, 50],  'color': 'rgba(240,192,64,0.10)'},
                            {'range': [50, 75],  'color': 'rgba(240,192,64,0.15)'},
                            {'range': [75, 100], 'color': 'rgba(0,229,160,0.10)'},
                            {'range': [100, 150],'color': 'rgba(0,229,160,0.18)'},
                        ],
                        'threshold': {'line': {'color': gauge_color, 'width': 3}, 'thickness': 0.8, 'value': roi}
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#7ecfb3', height=280, margin=dict(t=40, b=20, l=30, r=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Price Projection Bar Chart
            with chart_col2:
                st.markdown('<div class="section-badge">📈 Price Projection</div>', unsafe_allow_html=True)

                years = [2024, 2025, 2026]
                prices_sqft = [current_price, (current_price + predicted_price) / 2, predicted_price]
                total_prices = [p * land_area for p in prices_sqft]

                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[str(y) for y in years],
                    y=prices_sqft,
                    marker=dict(
                        color=['rgba(0,229,160,0.4)', 'rgba(0,229,160,0.65)', 'rgba(0,229,160,1)'],
                        line=dict(color='rgba(0,229,160,0.8)', width=1.5)
                    ),
                    text=[f"₹{p:,.0f}" for p in prices_sqft],
                    textposition='outside',
                    textfont=dict(color='#00e5a0', size=11, family='DM Mono'),
                    hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}/sqft<extra></extra>"
                ))
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, tickfont=dict(color='#7ecfb3')),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,229,160,0.08)', tickfont=dict(color='#7ecfb3'),
                               title=dict(text="₹ per sqft", font=dict(color='#7ecfb3'))),
                    height=280, margin=dict(t=40, b=20, l=50, r=30),
                    font_color='#7ecfb3',
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── 4. MAP ───────────────────────────────────────
            st.markdown('<div class="section-badge">🗺️ Property Location Map</div>', unsafe_allow_html=True)

            try:
                import folium
                from streamlit_folium import st_folium

                lat, lon = get_city_coords(city)
                m = folium.Map(
                    location=[lat, lon], zoom_start=11,
                    tiles="CartoDB dark_matter"
                )
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(
                        f"""<div style='font-family:sans-serif;min-width:180px;'>
                            <b>📍 {locality}</b><br>
                            {city}, {state}<br><hr>
                            <b>Current:</b> ₹{current_price:,.0f}/sqft<br>
                            <b>Predicted:</b> ₹{predicted_price:,.0f}/sqft<br>
                            <b>Total Value:</b> {format_inr(total_predicted)}<br>
                            <b>ROI:</b> {roi:.2f}%
                        </div>""",
                        max_width=250
                    ),
                    tooltip=f"📍 {locality}, {city}",
                    icon=folium.Icon(color="green", icon="home", prefix="fa")
                ).add_to(m)

                folium.Circle(
                    [lat, lon], radius=dist_city * 1000,
                    color="#00e5a0", fill=True, fill_opacity=0.05,
                    tooltip=f"City center radius: {dist_city} km"
                ).add_to(m)

                st_folium(m, width="100%", height=380, returned_objects=[])

            except ImportError:
                coords = get_city_coords(city)
                fig_map = go.Figure(go.Scattermapbox(
                    lat=[coords[0]], lon=[coords[1]],
                    mode='markers',
                    marker=dict(size=18, color='#00e5a0', opacity=0.9),
                    text=[f"{locality}, {city}"],
                    hovertemplate=(
                        f"<b>📍 {locality}, {city}</b><br>"
                        f"Current: ₹{current_price:,.0f}/sqft<br>"
                        f"Predicted: ₹{predicted_price:,.0f}/sqft<br>"
                        f"ROI: {roi:.2f}%<extra></extra>"
                    )
                ))
                fig_map.update_layout(
                    mapbox=dict(style="carto-darkmatter", center=dict(lat=coords[0], lon=coords[1]), zoom=10),
                    paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0, l=0, r=0), height=380
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption("💡 For a richer map, install: `pip install folium streamlit-folium`")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 5. INVESTMENT SUGGESTION ─────────────────────
            st.markdown('<div class="section-badge">📍 Investment Site Suggestion</div>', unsafe_allow_html=True)

            if roi > 100:
                st.success("🟢 Excellent Investment Site")
                buyer_text = f"""This property in {city}, {state} represents an exceptional investment opportunity 
                with projected returns exceeding 100%. The {zoning} zoning classification strengthens 
                the asset's versatility, offering wide-ranging development options aligned with current market demand. 
                Properties of this caliber in {city} are increasingly sought after by institutional investors. 
                We strongly recommend securing this asset at the earliest opportunity."""
                agent_text = f"""This listing in {city}, {state} presents one of the strongest investment cases available. 
                The {zoning} zoning broadens the eligible buyer pool significantly — attractive to residential developers, 
                commercial investors, and individual buyers alike. Prioritize this property in client presentations. 
                The high ROI and location advantage make this a compelling, time-sensitive opportunity."""

            elif roi > 75:
                st.info("🔵 Good Investment Site")
                buyer_text = f"""This property in {city}, {state} offers a compelling investment proposition backed by 
                strong projected returns. The {zoning} zoning supports flexible development options. {city} continues 
                to demonstrate consistent growth indicators, making this well-suited for investors seeking reliable 
                appreciation over a medium-term horizon."""
                agent_text = f"""This {zoning}-zoned property in {city}, {state} is well-supported by market data. 
                Present the growth trajectory of {city} and permissible-use advantages of {zoning} zoning as the 
                primary value drivers when engaging serious buyers."""

            elif roi > 50:
                st.warning("🟡 Moderate Investment Site")
                buyer_text = f"""This {zoning}-zoned property in {city}, {state} presents a moderate investment 
                opportunity suitable for buyers with a long-term perspective. {city} is on a steady development 
                trajectory and the {zoning} zoning allows for practical utilization during the holding period — 
                a prudent choice for capital preservation with measured appreciation over 5–7 years."""
                agent_text = f"""This property in {city}, {state} is best positioned for conservative or owner-use 
                buyers. Focus on {city}'s long-term development roadmap and the {zoning} zoning's practical 
                utility to build a credible, transparent investment case."""

            elif roi >= 25:
                st.warning("🟠 Below Average Investment Site")
                buyer_text = f"""This {zoning}-zoned property in {city}, {state} carries a below-average return 
                profile and is best suited for investors with extended holding capacity. A thorough assessment of 
                future infrastructure plans and regulatory developments specific to {city} is strongly advised 
                before making a commitment."""
                agent_text = f"""This property in {city}, {state} requires careful client qualification. The 
                current return profile limits suitability to long-horizon investors with low liquidity requirements. 
                Ensure complete transparency regarding realistic timelines and {city}'s market dynamics."""

            else:
                st.error("🔴 High Risk Investment Site")
                buyer_text = f"""This {zoning}-zoned property in {city}, {state} is currently classified as a 
                high-risk investment. Significant caution is advised before proceeding."""
                agent_text = f"""Professional integrity requires full and transparent disclosure of the risks 
                associated with this {zoning}-zoned property in {city}, {state}."""

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**👤 For the Buyer**")
                st.markdown(f'<div class="info-card">{buyer_text}</div>', unsafe_allow_html=True)
            with bc2:
                st.markdown("**🤝 For the Agent**")
                st.markdown(f'<div class="info-card">{agent_text}</div>', unsafe_allow_html=True)

            # ── 6. DOWNLOAD REPORT ────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            report = f"""
LANDSENSE — PROPERTY VALUATION REPORT
======================================
Location       : {locality}, {city}, {state}
Zoning         : {zoning}
City Tier      : {city_tier}
Land Area      : {land_area:,.2f} sqft

PRICE PER SQFT
--------------
Current        : ₹{current_price:,.2f}
Predicted 2026 : ₹{predicted_price:,.2f}
Difference     : ₹{diff:,.2f}
ROI            : {roi:.2f}%

TOTAL LAND VALUATION
--------------------
Current Total  : {format_inr(total_current)} (₹{total_current:,.0f})
Predicted Total: {format_inr(total_predicted)} (₹{total_predicted:,.0f})
Projected Gain : {format_inr(abs(total_diff))} (₹{total_diff:,.0f})

PROPERTY DETAILS
----------------
Distance to City Center  : {dist_city} km
Distance to Highway      : {dist_highway} km
Distance to Transport    : {dist_transport} km
Distance to Amenities    : {dist_amenities} km
Historical Growth        : {historical_growth}%
Population Growth        : {population_growth}%
Road Quality Score       : {road_quality}
Utility Access           : {"Yes" if utility_access else "No"}
Govt Development Plan    : {"Yes" if govt_dev_plan else "No"}
Flood Risk               : {"Yes" if flood_risk else "No"}

Generated by LandSense AI
"""
            st.download_button(
                "📄 Download Valuation Report",
                data=report,
                file_name=f"LandSense_{city}_{locality}.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error("⚠️ Error in prediction")
            st.exception(e)
