import streamlit as st

if "df" not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None

st.subheader("Welcome to the Simplify AI Platform!")

st.write("""
Explore and interact with machine learning models using the Scikit-learn (`sklearn`) library and Neural Networks with `PyTorch`. This platform is tailored for both binary and multi-class classification tasks.

### Features:

**Data Preprocessing**: Equip yourself with a suite of preprocessing tools. Visualize data distributions, detect class imbalances, and normalize for optimal performance.

**Model Selection**: Access a comprehensive library of algorithms ranging from regression and classification to clustering.

**Data Input**: Seamlessly upload datasets and define your features and target variables.

**Model Training & Evaluation**: Effortlessly train models and evaluate their performance with intuitive metrics and interactive visualizations.

**Hyperparameter Tuning**: Refine your models with a user-friendly slider-based interface for hyperparameter adjustments.

**Dive in, upload your data, and embark on your machine learning journey!**
""")

