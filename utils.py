import pandas as pd
import joblib
import streamlit as st

@st.cache_resource
def load_joblib(path: str):
    return joblib.load(path)

@st.cache_data
def load_products(path: str):
    return pd.read_excel(path, engine="openpyxl")
