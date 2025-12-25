import streamlit as st
from model import predict_quality, accuracy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Wine Quality Analyzer 🍷 ")
st.write("Predict wine quality using machine learning")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Quality Prediction", "Data Analysis"])

if page == "Quality Prediction":
    st.header("Quality Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.0, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.9, 15.0, 2.5, 0.1)
        chlorides = st.slider("Chlorides", 0.01, 0.6, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 15.0, 1.0)

    with col2:
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 46.0, 1.0)
        density = st.slider("Density", 0.990, 1.004, 0.997, 0.001)
        ph = st.slider("pH", 2.7, 4.0, 3.3, 0.01)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, 0.01)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5, 0.1)

    if st.button("Predict Quality"):
        prediction = predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol)
        st.success(f"Predicted: {prediction}")

elif page == "Data Analysis":
    st.header("Data Analysis")

    df = pd.read_csv('winequality.csv')

    # Show model accuracy in data analysis section
    st.metric("Model Accuracy", f"{accuracy:.0%}")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # Key metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Wines", len(df))
    with col2:
        st.metric("Average Quality", f"{df['quality'].mean():.1f}")

    # Visualizations using matplotlib
    st.subheader("Quality Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['quality'], bins=6, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Quality')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Wine Quality')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Scatter plot for alcohol vs quality
    st.subheader("Alcohol vs Quality")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['alcohol'], df['quality'], alpha=0.6, color='blue')
    ax.set_xlabel('Alcohol')
    ax.set_ylabel('Quality')
    ax.set_title('Relationship between Alcohol and Quality')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # pH vs quality
    st.subheader("pH vs Quality")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['pH'], df['quality'], alpha=0.6, color='green')
    ax.set_xlabel('pH')
    ax.set_ylabel('Quality')
    ax.set_title('Relationship between pH and Quality')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

