import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
from model import load_and_merge_data, train_model

st.title("Pokémon TCG Price Predictor")

# Carga de datos
@st.cache_data
def get_data():
    df_stats = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "abcsds/pokemon", "Pokemon.csv")
    df_precios = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "kanchana1990/e-commerce-pokmon-card-pricing-data", "pokemon_cards_ultimate_2026.csv")
    return load_and_merge_data(df_precios, df_stats)

try:
    df = get_data()
    st.success(f"Dataset cargado correctamente: {len(df):,} registros.")
except Exception as e:
    st.error(f"Error al cargar los datasets: {e}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Configuración del modelo
st.subheader("Configuración del Modelo")
n_estimators = st.slider("Número de árboles (n_estimators)", 50, 300, 150, 25)
max_depth = st.slider("Profundidad máxima del árbol (max_depth)", 3, 20, 10)

# Botón de ejecución
if st.button("Entrenar modelo"):
    with st.spinner("Entrenando modelo Random Forest..."):
        metrics, importance, comparison = train_model(df, n_estimators, max_depth)

    st.subheader("Evaluación")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (Log)", f"{metrics['r2_log']:.4f}")
    m2.metric("R² (USD)", f"{metrics['r2_usd']:.4f}")
    m3.metric("MAE (USD)", f"${metrics['mae_usd']:.2f}")

    # Visualización: Real vs Predicho
    st.subheader("Visualización: Real vs Predicho")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(comparison["Precio real (USD)"], comparison["Precio predicho (USD)"], alpha=0.5)
    
    min_v = min(comparison["Precio real (USD)"].min(), comparison["Precio predicho (USD)"].min())
    max_v = max(comparison["Precio real (USD)"].max(), comparison["Precio predicho (USD)"].max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)
    ax.set_xlabel("Precio real (USD)")
    ax.set_ylabel("Precio predicho (USD)")
    st.pyplot(fig)
    plt.close(fig)

    # Visualización: Importancia de características
    st.subheader("Importancia de Características")
    top10 = importance.head(10).sort_values("Importancia", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top10, x="Importancia", y="Característica", ax=ax)
    st.pyplot(fig)
    plt.close(fig)