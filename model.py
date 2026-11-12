import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_merge_data(df_precios, df_stats):
    """Limpia y une los datasets de estadísticas y precios."""
    df_stats["name_clean"] = df_stats["Name"].astype(str).str.lower().str.strip()
    df_precios["name_clean"] = df_precios["pokemon_name"].astype(str).str.lower().str.strip()
    
    df = pd.merge(df_precios, df_stats, on="name_clean", how="inner")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    return df

def train_model(df, n_estimators=150, max_depth=10):
    """Preprocesa los datos, entrena el modelo y calcula métricas e importancia."""
    df_model = df.dropna(subset=["price_usd"]).copy()
    
    columnas_excluir = ["price", "price_usd", "price_tier_usd", "currency", "Name", "name_clean", "#"]
    features = [col for col in df_model.columns if col not in columnas_excluir]

    X = df_model[features]
    y = np.log1p(df_model["price_usd"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

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
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    pred_usd = np.maximum(0, np.expm1(pred_log))
    test_usd = np.expm1(y_test)

    metrics = {
        "r2_log": r2_score(y_test, pred_log),
        "r2_usd": r2_score(test_usd, pred_usd),
        "mae_usd": mean_absolute_error(test_usd, pred_usd)
    }

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

    return metrics, importance, comparison