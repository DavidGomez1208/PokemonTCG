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

st.title("Análisis de precios de Pokémon TCG")
st.write(
)


@st.cache_data(show_spinner="Descargando y preparando los datasets...")
def cargar_datos():
    # Dataset con estadísticas de los Pokémon
    df_stats = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "abcsds/pokemon",
        "Pokemon.csv"
    )

    # Dataset con precios de cartas
    df_precios = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "kanchana1990/e-commerce-pokmon-card-pricing-data",
        "pokemon_cards_ultimate_2026.csv"
    )

    # Normalizar nombres para poder relacionar ambos datasets
    df_stats["name_clean"] = df_stats["Name"].astype(str).str.lower().str.strip()
    df_precios["name_clean"] = (
        df_precios["pokemon_name"].astype(str).str.lower().str.strip()
    )

    # Unir los datasets
    df = pd.merge(df_precios, df_stats, on="name_clean", how="inner")

    # Asegurar que el precio sea numérico
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    return df


@st.cache_data(show_spinner="Entrenando el modelo...")
def entrenar_modelo(df):
    # Eliminar filas sin precio
    df_model = df.dropna(subset=["price_usd"]).copy()

    # Excluir variables que revelan directamente el precio o identifican el registro
    columnas_excluir = [
        "price", "price_usd", "price_tier_usd", "currency",
        "Name", "name_clean", "#"
    ]
    features = [
        col for col in df_model.columns if col not in columnas_excluir
    ]

    X = df_model[features]
    y = np.log1p(df_model["price_usd"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detectar columnas numéricas y categóricas
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # Predicciones y regreso a la escala original (USD)
    pred_log = model.predict(X_test)
    pred_usd = np.maximum(0, np.expm1(pred_log))
    test_usd = np.expm1(y_test)

    metrics = {
        "r2_log": r2_score(y_test, pred_log),
        "r2_usd": r2_score(test_usd, pred_usd),
        "mae_usd": mean_absolute_error(test_usd, pred_usd)
    }

    # Importancia de las variables
    encoder = (
        model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    all_features = numeric_cols + encoded_cat_cols

    importance = pd.DataFrame({
        "Característica": all_features,
        "Importancia": model.named_steps["regressor"].feature_importances_
    }).sort_values("Importancia", ascending=False)

    comparison = pd.DataFrame({
        "Precio real (USD)": test_usd,
        "Precio predicho (USD)": pred_usd
    })

    return model, metrics, importance, comparison


try:
    df_completo = cargar_datos()
except Exception as error:
    st.error("No fue posible cargar los datasets de Kaggle.")
    st.write(
        "Comprueba tu conexión a internet y que KaggleHub pueda acceder "
        "a los datasets."
    )
    st.exception(error)
    st.stop()


if df_completo.empty:
    st.warning("La unión de los datasets no produjo registros.")
    st.stop()

st.success(f"Datos cargados correctamente: {len(df_completo):,} registros.")

# Métricas generales
col1, col2, col3 = st.columns(3)
col1.metric("Registros", f"{len(df_completo):,}")
col2.metric(
    "Pokémon diferentes",
    f"{df_completo['name_clean'].nunique():,}"
)
col3.metric(
    "Precio promedio",
    f"${df_completo['price_usd'].mean():,.2f} USD"
)

with st.expander("Ver datos y columnas del dataset"):
    st.dataframe(df_completo.head(20), use_container_width=True)
    st.write("Información de las columnas:")
    st.write(df_completo.dtypes.astype(str).to_frame("Tipo de dato"))


tab1, tab2, tab3, tab4 = st.tabs([
    "Exploración",
    "Correlaciones",
    "Modelo predictivo",
    "Importancia de variables"
])

with tab1:
    st.subheader("Precio promedio según el tipo principal")

    if {"Type 1", "price_usd"}.issubset(df_completo.columns):
        resumen_tipos = (
            df_completo.dropna(subset=["Type 1", "price_usd"])
            .groupby("Type 1", as_index=False)["price_usd"]
            .mean()
            .sort_values("price_usd", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=resumen_tipos,
            x="Type 1",
            y="price_usd",
            ax=ax
        )
        ax.set_xlabel("Tipo principal")
        ax.set_ylabel("Precio promedio (USD)")
        ax.set_title("Precio promedio de las cartas por tipo de Pokémon")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No se encontraron las columnas necesarias para esta gráfica.")

    st.subheader("Resumen estadístico")
    columnas_resumen = [
        c for c in ["price_usd", "HP", "Attack", "Defense", "Total"]
        if c in df_completo.columns
    ]
    st.dataframe(df_completo[columnas_resumen].describe(), use_container_width=True)


with tab2:
    st.subheader("Correlación entre estadísticas y precio")

    columnas_correlacion = [
        c for c in [
            "price_usd", "HP", "Attack", "Defense",
            "Sp. Atk", "Sp. Def", "Speed", "Total"
        ]
        if c in df_completo.columns
    ]

    matriz = df_completo[columnas_correlacion].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matriz,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlación entre estadísticas de Pokémon y precio")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Una correlación cercana a 1 indica una relación positiva; "
        "cercana a -1, una relación negativa; y cercana a 0, una relación "
        "lineal débil. Correlación no significa causalidad."
    )


with tab3:
    st.subheader("Evaluación del modelo Random Forest")

    try:
        modelo, metricas, importancia, comparacion = entrenar_modelo(
            df_completo
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("R² (escala logarítmica)", f"{metricas['r2_log']:.4f}")
        m2.metric("R² (USD)", f"{metricas['r2_usd']:.4f}")
        m3.metric("Error absoluto medio", f"${metricas['mae_usd']:.2f}")

        st.write(
            "El MAE representa, en promedio, cuánto se alejan las "
            "predicciones del precio real, en dólares."
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            comparacion["Precio real (USD)"],
            comparacion["Precio predicho (USD)"],
            alpha=0.5
        )

        minimo = min(
            comparacion["Precio real (USD)"].min(),
            comparacion["Precio predicho (USD)"].min()
        )
        maximo = max(
            comparacion["Precio real (USD)"].max(),
            comparacion["Precio predicho (USD)"].max()
        )
        ax.plot([minimo, maximo], [minimo, maximo], "r--", linewidth=2)
        ax.set_xlabel("Precio real (USD)")
        ax.set_ylabel("Precio predicho (USD)")
        ax.set_title("Precios reales frente a precios predichos")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as error:
        st.error("Ocurrió un problema al entrenar el modelo.")
        st.exception(error)


with tab4:
    st.subheader("Las 10 características más importantes")

    try:
        # Se reutiliza el resultado almacenado en caché
        _, _, importancia, _ = entrenar_modelo(df_completo)

        top10 = importancia.head(10).sort_values(
            "Importancia", ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=top10,
            x="Importancia",
            y="Característica",
            ax=ax
        )
        ax.set_title("Top 10 de variables más importantes")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(importancia.head(10), use_container_width=True)

    except Exception as error:
        st.error("No fue posible calcular la importancia de las variables.")
        st.exception(error)

st.divider()
st.caption("Aplicación desarrollada con Python, Streamlit, Pandas y Scikit-learn.")
