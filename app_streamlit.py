# -*- coding: utf-8 -*-
import streamlit as st
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kagglehub import KaggleDatasetAdapter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.title("Precios de Cartas Pokémon TCG")
st.write("Análisis rápido de estadísticas y modelo predictivo de precios.")

@st.cache_data
def cargar_datos():
    df_stats = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "abcsds/pokemon", "Pokemon.csv")
    df_precios = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "kanchana1990/e-commerce-pokmon-card-pricing-data", "pokemon_cards_ultimate_2026.csv")
    
    df_stats["name_clean"] = df_stats["Name"].astype(str).str.lower().str.strip()
    df_precios["name_clean"] = df_precios["pokemon_name"].astype(str).str.lower().str.strip()
    
    df = pd.merge(df_precios, df_stats, on="name_clean", how="inner")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    return df

@st.cache_data
def entrenar_modelo(df):
    df_model = df.dropna(subset=["price_usd"]).copy()
    columnas_excluir = ["price", "price_usd", "price_tier_usd", "currency", "Name", "name_clean", "#"]
    features = [col for col in df_model.columns if col not in columnas_excluir]

    X = df_model[features]
    y = np.log1p(df_model["price_usd"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    pred_usd = np.maximum(0, np.expm1(pred_log))
    test_usd = np.expm1(y_test)

    metrics = {
        "r2_usd": r2_score(test_usd, pred_usd),
        "mae_usd": mean_absolute_error(test_usd, pred_usd)
    }

    encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    all_features = numeric_cols + encoder.get_feature_names_out(categorical_cols).tolist()

    importance = pd.DataFrame({
        "Característica": all_features,
        "Importancia": model.named_steps["regressor"].feature_importances_
    }).sort_values("Importancia", ascending=False)

    comparison = pd.DataFrame({
        "Precio real": test_usd,
        "Precio predicho": pred_usd
    })

    return metrics, importance, comparison

try:
    df_completo = cargar_datos()
    st.success(f"Datos cargados: {len(df_completo):,} registros.")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Métricas rápidas
c1, c2, c3 = st.columns(3)
c1.metric("Registros", f"{len(df_completo):,}")
c2.metric("Pokémon únicos", f"{df_completo['name_clean'].nunique():,}")
c3.metric("Precio promedio", f"${df_completo['price_usd'].mean():,.2f}")

st.divider()

# Organización en dos pestañas más limpias
tab1, tab2 = st.tabs(["📊 Exploración y Correlaciones", "🤖 Modelo Random Forest"])

with tab1:
    st.subheader("Precio por Tipo Principal")
    resumen_tipos = df_completo.dropna(subset=["Type 1", "price_usd"]).groupby("Type 1", as_index=False)["price_usd"].mean().sort_values("price_usd", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=resumen_tipos, x="Type 1", y="price_usd", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Matriz de Correlación")
    cols_corr = [c for c in ["price_usd", "HP", "Attack", "Defense", "Speed", "Total"] if c in df_completo.columns]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_completo[cols_corr].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Resultados del Modelo")
    metrics, importancia, comparacion = entrenar_modelo(df_completo)

    m1, m2 = st.columns(2)
    m1.metric("R² Score (USD)", f"{metrics['r2_usd']:.4f}")
    m2.metric("Error Absoluto Medio (MAE)", f"${metrics['mae_usd']:.2f}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("Real vs Predicho")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(comparacion["Precio real"], comparacion["Precio predicho"], alpha=0.4)
        min_v = min(comparacion["Precio real"].min(), comparacion["Precio predicho"].min())
        max_v = max(comparacion["Precio real"].max(), comparacion["Precio predicho"].max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--")
        st.pyplot(fig)
        plt.close(fig)

    with col_g2:
        st.write("Top Variables Importantes")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=importancia.head(8), x="Importancia", y="Característica", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
