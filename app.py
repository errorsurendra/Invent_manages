import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide"
)
# ==============================
# ZIDIO Header
# ==============================

col1, col2 = st.columns([1, 6])

with col1:
    logo_path = Path("assets/zidio_logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("**ZIDIO**")

with col2:
    st.markdown("""
    <h1 style='margin-bottom:0;color:#1E88E5;'>
        ZIDIO DEVELOPMENT
    </h1>
    <h4 style='margin-top:0;color:gray;'>
        Project FORESIGHT – AI-Powered Demand Forecasting & Inventory Intelligence
    </h4>
    """, unsafe_allow_html=True)

st.markdown("---")

st.title("📦 Project FORESIGHT")
st.subheader("AI-Powered Demand Forecasting & Inventory Intelligence")

# -----------------------------
# LOAD DATA
# -----------------------------

def load_csv(file_path, columns=None):
    path = Path(file_path)
    if not path.exists():
        st.warning(f"Missing file: {path.name}. Please add it to the app folder.")
        return pd.DataFrame(columns=columns if columns is not None else [])
    return pd.read_csv(path)

sales = load_csv(
    "featured_sales.csv",
    columns=["Date", "Amount", "Boxes_Shipped", "Product", "Country"]
)
inventory = load_csv(
    "inventory_report.csv",
    columns=["Product", "Status", "Stock_Level", "Reorder_Level"]
)

# Load trained model (if available)
model_file = Path("models/demand_model.pkl")
if model_file.exists():
    try:
        model = joblib.load(model_file)
    except Exception as exc:
        st.warning(f"Unable to load model: {exc}")
        model = None
else:
    model = None

# -----------------------------
# KPI CARDS
# -----------------------------
total_revenue = sales["Amount"].sum()
total_boxes = sales["Boxes_Shipped"].sum()
total_products = sales["Product"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Revenue", f"${total_revenue:,.0f}")
col2.metric("Boxes Sold", int(total_boxes))
col3.metric("Products", total_products)

st.divider()

# -----------------------------
# PRODUCT FILTER
# -----------------------------
product = st.selectbox(
    "Select Product",
    ["All"] + sorted(sales["Product"].unique().tolist())
)

filtered_sales = sales.copy()

if product != "All":
    filtered_sales = sales[sales["Product"] == product]

# -----------------------------
# REVENUE TREND
# -----------------------------
st.header("📈 Revenue Trend")

filtered_sales["Date"] = pd.to_datetime(filtered_sales["Date"])

daily = (
    filtered_sales.groupby("Date")["Amount"]
    .sum()
    .reset_index()
)

fig = px.line(
    daily,
    x="Date",
    y="Amount",
    title="Revenue Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MONTHLY SALES
# -----------------------------
st.header("📊 Monthly Sales")

filtered_sales["Month"] = filtered_sales["Date"].dt.strftime("%b")

monthly = (
    filtered_sales.groupby("Month")["Amount"]
    .sum()
    .reset_index()
)

fig = px.bar(
    monthly,
    x="Month",
    y="Amount",
    title="Monthly Revenue"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PRODUCT REVENUE PIE
# -----------------------------
st.header("🥧 Product Revenue")

pie = (
    sales.groupby("Product")["Amount"]
    .sum()
    .reset_index()
)

fig = px.pie(
    pie,
    names="Product",
    values="Amount"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# COUNTRY SALES
# -----------------------------
st.header("🌍 Country Revenue")

country = (
    sales.groupby("Country")["Amount"]
    .sum()
    .reset_index()
)

fig = px.bar(
    country,
    x="Country",
    y="Amount",
    color="Country"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INVENTORY STATUS
# -----------------------------
st.header("📦 Inventory Status")

st.dataframe(inventory, use_container_width=True)

# -----------------------------
# STOCK ALERTS
# -----------------------------
st.header("⚠️ Inventory Alerts")

reorder = inventory[inventory["Status"] == "Reorder"]
overstock = inventory[inventory["Status"] == "Overstock"]

c1, c2 = st.columns(2)

with c1:
    st.error("Products Needing Reorder")
    st.dataframe(reorder)

with c2:
    st.warning("Overstock Products")
    st.dataframe(overstock)

# -----------------------------
# AI DEMAND PREDICTION
# -----------------------------
st.header("🤖 AI Demand Prediction")

if model is not None:

    amount = st.number_input("Amount", value=4000.0)
    product_id = st.number_input("Product ID", value=1)
    country_id = st.number_input("Country ID", value=1)
    salesperson_id = st.number_input("Salesperson ID", value=1)
    month = st.slider("Month", 1, 12, 7)

    if st.button("Predict Demand"):

        sample = pd.DataFrame({
            "Amount":[amount],
            "Product_ID":[product_id],
            "Country_ID":[country_id],
            "Salesperson_ID":[salesperson_id],
            "Year":[2022],
            "Month":[month],
            "Day":[1],
            "Day_of_Week":[2],
            "Quarter":[2],
            "Weekend":[0],
            "Revenue_per_Box":[20],
            "Lag_1":[150],
            "Lag_7":[160],
            "Rolling_7":[170],
            "Rolling_30":[180]
        })

        prediction = model.predict(sample)

        st.success(
            f"Predicted Boxes Shipped : {prediction[0]:.0f}"
        )

else:
    st.info("Train the model first.")

# -----------------------------
# DOWNLOAD CSV
# -----------------------------
st.header("📥 Download Inventory Report")

csv = inventory.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "inventory_report.csv",
    "text/csv"
)

# -----------------------------
# RAW DATA
# -----------------------------
st.header("📋 Sales Data")

st.dataframe(filtered_sales, use_container_width=True)