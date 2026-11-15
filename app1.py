import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# Import from src
from src.data_loader import load_data
from src.predict import load_model, make_prediction

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (MODERN UI)
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.kpi-card {
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    transition: 0.3s;
}
.kpi-card:hover {
    transform: scale(1.05);
}

.kpi-title {
    font-size: 14px;
    opacity: 0.8;
}
.kpi-value {
    font-size: 32px;
    font-weight: bold;
}

.section-title {
    font-size: 22px;
    margin-top: 20px;
    color: #ffffff;
}

.stNumberInput input {
        background-color: #f0f8ff;
        border-radius: 8px;
    }

 div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = load_data()

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model()

# -------------------------------
# TITLE
# -------------------------------
st.title("🚀 Customer Churn Intelligence Dashboard")

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

cat_cols = df.select_dtypes(include="object").columns.tolist()

filters = {}
for col in cat_cols:
    if col == "Churn":
        continue
    else:
        values = df[col].dropna().unique()
        selected = st.sidebar.multiselect(col, values, default=values)
        filters[col] = selected

filtered_df = df.copy()
for col, vals in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(vals)]

# -------------------------------
# KPI SECTION
# -------------------------------
st.markdown("### 📊 Key Metrics")

churn_col = [c for c in df.columns if "churn" in c.lower()][0]

total_customers = len(filtered_df)
churned = filtered_df[filtered_df[churn_col] == "Yes"].shape[0]
churn_rate = (churned / total_customers) * 100 if total_customers > 0 else 0
missing = filtered_df.isnull().sum().sum()

col1, col2, col3, col4 = st.columns(4)

def kpi(title, value):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

col1.markdown(kpi("Total Customers", total_customers), unsafe_allow_html=True)
col2.markdown(kpi("Churned Customers", churned), unsafe_allow_html=True)
col3.markdown(kpi("Churn Rate", f"{churn_rate:.2f}%"), unsafe_allow_html=True)
col4.markdown(kpi("Missing Values", missing), unsafe_allow_html=True)

# -------------------------------
# CHARTS
# -------------------------------
st.markdown("### 📈 Customer Insights")

c1, c2 = st.columns(2)

# Pie Chart
churn_counts = filtered_df[churn_col].value_counts().reset_index()
churn_counts.columns = ["Churn", "Count"]

fig1 = px.pie(
    churn_counts,
    names="Churn",
    values="Count",
    color="Churn",
    color_discrete_map={"Yes": "#ff4b4b", "No": "#2ecc71"},
    title="Churn Distribution"
)
c1.plotly_chart(fig1, use_container_width=True)

# Contract Chart
if "Contract" in filtered_df.columns:
    fig2 = px.histogram(
        filtered_df,
        x="Contract",
        color=churn_col,
        barmode="group",
        color_discrete_map={"Yes": "#ff4b4b", "No": "#2ecc71"},
        title="Churn by Contract Type"
    )
    c2.plotly_chart(fig2, use_container_width=True)

# Row 2
c3, c4 = st.columns(2)

if "MonthlyCharges" in filtered_df.columns:
    fig3 = px.histogram(
        filtered_df,
        x="MonthlyCharges",
        nbins=30,
        color_discrete_sequence=["#3498db"],
        title="Monthly Charges Distribution"
    )
    c3.plotly_chart(fig3, use_container_width=True)

if "Tenure" in filtered_df.columns:
    fig4 = px.scatter(
        filtered_df,
        x="Tenure",
        y="MonthlyCharges",
        color=churn_col,
        color_discrete_map={"Yes": "#ff4b4b", "No": "#2ecc71"},
        title="Tenure vs Monthly Charges"
    )
    c4.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# PREDICTION SECTION
# -------------------------------
st.markdown("### 🤖 Churn Prediction")

input_data = {}

cols = st.columns(3)
col_index = 0 
for i, col in enumerate(df.columns):

    if col in ["Churn", "CustomerID"]:   # 👈 skip both
        continue
    current_col = cols[col_index % 3]

    if col in ["Age", "Tenure"]:
        val = current_col.number_input(
            col,
            min_value=0,
            max_value=100,
            value=int(df[col].mean()),   # 👈 remove decimal
            step=1                       # 👈 integer step
        )
    # ✅ NUMERIC COLUMNS
    elif df[col].dtype in ['int64', 'float64']:
        val = current_col.number_input(
            col,
            value=float(df[col].mean())
        )

    # ✅ CATEGORICAL COLUMNS
    else:
        val = current_col.selectbox(
            col,
            df[col].dropna().unique()
        )

    input_data[col] = val
    col_index += 1  
# Convert to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("🔮 Predict"):
    pred, prob = make_prediction(model, input_df)

    if pred == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk ({1 - prob:.2%})")

# -------------------------------
# DATA TABLE
# -------------------------------
st.markdown("### 📄 Customer Data")
st.dataframe(filtered_df.head(100))