import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(
    page_title="CustomerSeg AI",
    page_icon="👥",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

.big-title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #2563EB;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

.section-title {
    color: #2563EB;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model & Data
# ---------------------------

model = joblib.load("customer_segmentation_model.pkl")

df = pd.read_csv("Mall_Customers.csv")

X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

df["Cluster"] = model.predict(X)

# ---------------------------
# Hero Section
# ---------------------------

st.markdown("""
<div class='big-title'>
👥 CustomerSeg AI
</div>

<div class='subtitle'>
AI-Powered Customer Segmentation Dashboard
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------
# KPI Cards
# ---------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    len(df)
)

col2.metric(
    "Customer Segments",
    5
)

col3.metric(
    "Avg Income",
    f"${df['Annual Income (k$)'].mean():.0f}k"
)

col4.metric(
    "Avg Spending Score",
    f"{df['Spending Score (1-100)'].mean():.0f}"
)

st.divider()

# ---------------------------
# About App
# ---------------------------

st.markdown("""
## 📌 About the App

CustomerSeg AI uses **K-Means Clustering** to automatically identify groups of customers based on their income and spending behavior.

Businesses can use these insights to:

- Create targeted marketing campaigns
- Improve customer retention
- Increase sales
- Identify premium customer groups
""")

st.divider()

# ---------------------------
# Visualization
# ---------------------------

st.markdown("## 📈 Customer Segmentation Analysis")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    size="Age",
    hover_data=["Gender"],
    title="Customer Segments"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------
# Segment Distribution
# ---------------------------

st.markdown("## 🥧 Customer Segment Distribution")

segment_counts = df["Cluster"].value_counts()

fig2 = px.pie(
    values=segment_counts.values,
    names=segment_counts.index,
    title="Distribution of Customer Segments"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.divider()

# ---------------------------
# Customer Predictor
# ---------------------------

st.markdown("## 🎯 Customer Segment Predictor")

col1, col2 = st.columns(2)

with col1:
    income = st.slider(
        "Annual Income (k$)",
        10,
        150,
        50
    )

with col2:
    score = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

if st.button("Find Customer Segment"):

    cluster = model.predict(
        [[income, score]]
    )[0]

    segment_names = {
        0: "Budget Customers",
        1: "Premium Customers",
        2: "Average Customers",
        3: "High Value Customers",
        4: "Potential Loyal Customers"
    }

    st.success(
        f"🎯 Predicted Segment: {segment_names.get(cluster, 'Customer Segment')}"
    )

st.divider()

# ---------------------------
# Business Insights
# ---------------------------

st.markdown("""
## 💡 Business Insights

### Premium Customers
High income and high spending customers who are ideal targets for premium products.

### Budget Customers
Price-sensitive customers who respond well to discounts and promotions.

### Average Customers
Regular customers with moderate purchasing behavior.

### High Value Customers
Customers with strong purchasing power and excellent growth potential.

### Potential Loyal Customers
Customers who can be converted into long-term loyal customers through personalized engagement.
""")

st.divider()

# ---------------------------
# AI Model Information
# ---------------------------

st.markdown("""
## 🤖 AI Model Information

| Feature | Value |
|----------|----------|
| Algorithm | K-Means Clustering |
| Number of Clusters | 5 |
| Dataset | Mall Customer Segmentation |
| Framework | Scikit-Learn |
| Deployment | Streamlit |
""")

st.divider()

# ---------------------------
# Footer
# ---------------------------

st.markdown("""
<center>

Built with ❤️ using Python • Scikit-Learn • Plotly • Streamlit

</center>
""", unsafe_allow_html=True)