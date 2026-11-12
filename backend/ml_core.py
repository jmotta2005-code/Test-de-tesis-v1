"""Lógica de ML pura: sin FastAPI ni Streamlit."""
import io

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLF_SCORING = {
    "F1 macro": "f1_macro",
    "Balanced accuracy": "balanced_accuracy",
    "Accuracy": "accuracy",
    "F1 ponderado": "f1_weighted",
    "Precision macro": "precision_macro",
    "Recall macro": "recall_macro",
}
REG_SCORING = {
    "R²": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
}
ESPACIO_ALEATORIO = {
    "rf__n_estimators": [100, 200, 300, 500],
    "rf__max_depth": [None, 5, 10, 15, 20, 30],
    "rf__min_samples_split": [2, 5, 10, 20],
    "rf__min_samples_leaf": [1, 2, 4, 8],
    "rf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
}
ESPACIO_REJILLA = {
    "rf__n_estimators": [100, 300],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_leaf": [1, 3, 5],
    "rf__max_features": ["sqrt", 0.5],
}


# ------------------------------------------------------------------ lectura
def hojas_excel(contenido: bytes, nombre: str) -> list:
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.ExcelFile(io.BytesIO(contenido)).sheet_names
    return []


def leer_archivo(contenido: bytes, nombre: str, hoja=None) -> pd.DataFrame:
    n = nombre.lower()
    buf = io.BytesIO(contenido)
    if n.endswith(".csv"):
        return pd.read_csv(buf, sep=None, engine="python")
    if n.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(buf)
        return xls.parse(hoja or xls.sheet_names[0])
    raise ValueError(f"Formato no soportado: {nombre}")


# -------------------------------------------------------------- preparación
def es_clasificacion(y: pd.Series) -> bool:
    if y.dtype == "object" or str(y.dtype) in ("category", "bool"):
        return True
    return bool(pd.api.types.is_integer_dtype(y) and y.nunique() <= 10)


def preparar_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(object)
        elif pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_extension_array_dtype(s):
                X[c] = s.astype("float64")
        else:
            X[c] = s.astype(object).where(s.notna(), np.nan)
    return X


def auto_exclusiones(datos: pd.DataFrame, columnas: list) -> dict:
    out, n = {}, len(datos)
    for c in columnas:
        s = datos[c]
        nun = s.nunique(dropna=True)
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = "es de tipo fecha/hora"
        elif nun <= 1:
            out[c] = "es constante"
        elif not pd.api.types.is_numeric_dtype(s) and n > 20 and nun / n > 0.9:
            out[c] = "parece un ID o texto libre (casi todos los valores únicos)"
    return out


def preparar_datos(df_x: pd.DataFrame, df_y: pd.DataFrame, o: dict):
    """Une X e y, aplica exclusiones y devuelve (X, y, info)."""
    col_obj = o["col_objetivo"]
    col_id = o.get("col_id")
    if col_obj not in df_y.columns:
        raise ValueError(f"La columna objetivo '{col_obj}' no está en el archivo de target.")

    if col_id:
        extra_y = [c for c in df_y.columns
                   if c not in (col_id, col_obj) and c not in df_x.columns]
        datos = df_x.merge(df_y[[col_id, col_obj] + extra_y], on=col_id, how="inner")
        if datos.empty:
            raise ValueError("La unión por ID no produjo filas. Revisa que los valores coincidan.")
    else:
        if len(df_x) != len(df_y):
            raise ValueError(
                f"Los archivos tienen distinto número de filas ({len(df_x)} vs {len(df_y)}). "
                "Usa una columna ID o corrige los datos."
            )
        datos = df_x.reset_index(drop=True).copy()
        for c in df_y.columns:
            if c != col_obj and c not in datos.columns:
                datos[c] = df_y[c].reset_index(drop=True)
        datos[col_obj] = df_y[col_obj].reset_index(drop=True)

    info = {"filas_union": len(datos)}
    n_antes = len(datos)
    datos = datos.dropna(subset=[col_obj]).reset_index(drop=True)
    info["filas_sin_target"] = n_antes - len(datos)

    temporal = bool(o.get("temporal"))
    col_tiempo = o.get("col_tiempo")
    excluir_tiempo = temporal and o.get("excluir_tiempo", True)

    candidatas = [
        c for c in datos.columns
        if c not in (col_obj, col_id) and not (excluir_tiempo and c == col_tiempo)
    ]
    excluir = o.get("excluir", [])
    motivos = {}
    if o.get("auto_excl", True):
        motivos = auto_exclusiones(datos, [c for c in candidatas if c not in excluir])
    features = [c for c in candidatas if c not in excluir and c not in motivos]
    info["motivos"] = motivos

    umbral = o.get("umbral_card", 30)
    cat_alta = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(datos[c]) and datos[c].nunique(dropna=True) > umbral
    ]
    info["cat_alta"] = cat_alta
    if cat_alta and o.get("modo_card") == "excluir":
        features = [c for c in features if c not in cat_alta]
    if not features:
        raise ValueError("No quedan variables para entrenar.")

    info["n_dup"] = int(datos.duplicated(subset=features + [col_obj]).sum())
    if info["n_dup"] > 0 and o.get("quitar_dup", True):
        datos = datos.drop_duplicates(subset=features + [col_obj]).reset_index(drop=True)

    if temporal:
        if not col_tiempo:
            raise ValueError("Indica la columna de fecha / orden.")
        orden = datos[col_tiempo]
        if not pd.api.types.is_numeric_dtype(orden) and not pd.api.types.is_datetime64_any_dtype(orden):
            orden = pd.to_datetime(orden, errors="coerce")
        if orden.isna().mean() > 0.2:
            raise ValueError(f"No se pudo interpretar '{col_tiempo}' como fecha u orden (más de 20% inválido).")
        validas = orden.notna().to_numpy()
        datos = datos.loc[validas].reset_index(drop=True)
        orden = orden[validas].reset_index(drop=True)
        datos = datos.iloc[np.argsort(orden.to_numpy(), kind="stable")].reset_index(drop=True)

    X = preparar_X(datos[features])
    y = datos[col_obj]
    return X, y, info


def construir_preprocesador(X, escalar: bool, umbral_card: int, top_n: int):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    cat_alta = [c for c in cat_cols if X[c].nunique(dropna=True) > umbral_card]
    cat_baja = [c for c in cat_cols if c not in cat_alta]

    pasos_num = [("imp", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("sc", StandardScaler()))
    tr = [("num", Pipeline(pasos_num), num_cols)]
    if cat_baja:
        tr.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_baja))
    if cat_alta:
        tr.append(("cat_alta", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=top_n)),
        ]), cat_alta))
    return ColumnTransformer(tr, remainder="drop")


# ---------------------------------------------------------------- diagnóstico
def diagnostico_columnas(X: pd.DataFrame, umbral_card: int) -> pd.DataFrame:
    n = len(X)
    filas = []
    for c in X.columns:
        s = X[c]
        nun = int(s.nunique(dropna=True))
        numerica = pd.api.types.is_numeric_dtype(s)
        pct_out = np.nan
        if numerica and nun > 1:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                pct_out = round(100 * ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean(), 1)
        alertas = []
        if nun <= 1:
            alertas.append("constante")
        if s.isna().mean() > 0.3:
            alertas.append(">30% faltantes")
        if not numerica and nun > umbral_card:
            alertas.append("alta cardinalidad")
        if n > 20 and nun / n > 0.9 and not pd.api.types.is_float_dtype(s):
            alertas.append("posible ID")
        filas.append({
            "columna": c,
            "tipo": str(s.dtype),
            "% faltantes": round(100 * s.isna().mean(), 1),
            "valores únicos": nun,
            "% outliers (IQR)": pct_out,
            "alertas": ", ".join(alertas),
        })
    return pd.DataFrame(filas)


def escanear_fuga(X: pd.DataFrame, y: pd.Series, es_clf: bool, semilla: int = 42):
    if len(X) > 5000:
        idx = X.sample(5000, random_state=semilla).index
        X, y = X.loc[idx], y.loc[idx]

    if es_clf:
        modelo = DecisionTreeClassifier(max_depth=5, random_state=semilla)
        scoring = "balanced_accuracy"
        if y.value_counts().min() >= 3:
            cv = StratifiedKFold(3, shuffle=True, random_state=semilla)
        else:
            cv = KFold(3, shuffle=True, random_state=semilla)
    else:
        modelo = DecisionTreeRegressor(max_depth=5, random_state=semilla)
        scoring = "r2"
        cv = KFold(3, shuffle=True, random_state=semilla)

    filas = []
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_numeric_dtype(col):
            v = col.astype(float)
            v = v.fillna(v.median() if v.notna().any() else 0.0).to_numpy()
        else:
            v = pd.factorize(col.astype(str))[0]
        try:
            sc = cross_val_score(modelo, np.asarray(v).reshape(-1, 1), y.to_numpy(),
                                 cv=cv, scoring=scoring).mean()
        except Exception:  # noqa: BLE001
            sc = np.nan
        filas.append({"columna": c, "score con 1 sola variable": sc})

    df = pd.DataFrame(filas).sort_values("score con 1 sola variable", ascending=False)
    return df.reset_index(drop=True), ("balanced accuracy" if es_clf else "R²")


# ------------------------------------------------------------- entrenamiento
def metricas_clf(pipe, X_test, y_test, y_pred) -> dict:
    out = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
        "F1 macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Precision macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        proba = pipe.predict_proba(X_test)
        if proba.shape[1] == 2:
            out["ROC AUC"] = roc_auc_score(y_test, proba[:, 1])
        else:
            out["ROC AUC"] = roc_auc_score(y_test, proba, multi_class="ovr",
                                           average="macro", labels=pipe.classes_)
    except Exception:  # noqa: BLE001
        out["ROC AUC"] = np.nan
    return out


def entrenar_y_evaluar(X: pd.DataFrame, y: pd.Series, es_clf: bool, cfg: dict) -> dict:
    rs = cfg["random_state"]
    scoring = cfg["scoring"]
    cw = cfg["class_weight"]
    avisos = []

    if cfg["temporal"]:
        n_test = max(1, int(round(len(X) * cfg["test_size"])))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    else:
        estratificar = y if (es_clf and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["test_size"], random_state=rs, stratify=estratificar)

    n_splits = int(max(2, min(cfg["n_splits"], len(X_train) - 1)))
    if cfg["temporal"]:
        cv = TimeSeriesSplit(n_splits=n_splits)
    elif es_clf and y_train.value_counts().min() >= n_splits:
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=rs)
    else:
        if es_clf:
            avisos.append("Alguna clase tiene menos muestras que folds: se usó KFold sin estratificar.")
        cv = KFold(n_splits, shuffle=True, random_state=rs)

    rf_cls = RandomForestClassifier if es_clf else RandomForestRegressor
    prep_rf = construir_preprocesador(X_train, False, cfg["umbral_card"], cfg["top_n"])

    busqueda = None
    if cfg["search_mode"] == "Ninguna":
        params = dict(cfg["rf_params"], random_state=rs, n_jobs=-1)
        if es_clf and cw:
            params["class_weight"] = cw
        rf_pipe = Pipeline([("prep", prep_rf), ("rf", rf_cls(**params))])
    else:
        base_kwargs = dict(random_state=rs, n_jobs=1)
        if es_clf and cw:
            base_kwargs["class_weight"] = cw
        base = Pipeline([("prep", prep_rf), ("rf", rf_cls(**base_kwargs))])
        if cfg["search_mode"] == "Aleatoria":
            busqueda = RandomizedSearchCV(base, ESPACIO_ALEATORIO, n_iter=cfg["n_iter"], cv=cv,
                                          scoring=scoring, random_state=rs, n_jobs=-1,
                                          refit=True, error_score="raise")
        else:
            busqueda = GridSearchCV(base, ESPACIO_REJILLA, cv=cv, scoring=scoring,
                                    n_jobs=-1, refit=True, error_score="raise")
        busqueda.fit(X_train, y_train)
        rf_pipe = busqueda.best_estimator_
        rf_pipe.set_params(rf__n_jobs=-1)

    prep_lin = construir_preprocesador(X_train, True, cfg["umbral_card"], cfg["top_n"])
    prep_dummy = construir_preprocesador(X_train, False, cfg["umbral_card"], cfg["top_n"])
    if es_clf:
        simple = LogisticRegression(max_iter=2000, class_weight="balanced" if cw else None)
        trivial = DummyClassifier(strategy="most_frequent")
    else:
        simple = LinearRegression()
        trivial = DummyRegressor(strategy="mean")

    modelos = {
        "base": Pipeline([("prep", prep_dummy), ("modelo", trivial)]),
        "simple": Pipeline([("prep", prep_lin), ("modelo", simple)]),
        "rf": rf_pipe,
    }

    scorer = get_scorer(scoring)
    tabla = {}
    for nombre, pipe in modelos.items():
        r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                           return_train_score=True, n_jobs=1, error_score="raise")
        pipe.fit(X_train, y_train)
        tabla[nombre] = {
            "cv_mean": float(np.mean(r["test_score"])),
            "cv_std": float(np.std(r["test_score"])),
            "train_mean": float(np.mean(r["train_score"])),
            "test": float(scorer(pipe, X_test, y_test)),
        }

    y_pred = rf_pipe.predict(X_test)
    nombres = rf_pipe.named_steps["prep"].get_feature_names_out()
    imp_impureza = pd.Series(rf_pipe.named_steps["rf"].feature_importances_,
                             index=nombres).sort_values(ascending=False)

    imp_perm = None
    if cfg["usar_perm"]:
        Xs = X_test.sample(min(len(X_test), 2000), random_state=rs)
        ys = y_test.loc[Xs.index]
        pi = permutation_importance(rf_pipe, Xs, ys, scoring=scoring, n_repeats=10,
                                    random_state=rs, n_jobs=-1)
        imp_perm = pd.DataFrame({"media": pi.importances_mean, "std": pi.importances_std},
                                index=X.columns).sort_values("media", ascending=False)

    cv_top = None
    if busqueda is not None:
        cols = ["mean_test_score", "std_test_score"] + [
            c for c in busqueda.cv_results_ if c.startswith("param_")]
        cv_top = (pd.DataFrame(busqueda.cv_results_)[cols]
                  .sort_values("mean_test_score", ascending=False).head(10))
        cv_top.columns = [c.replace("param_rf__", "") for c in cv_top.columns]
        if scoring.startswith("neg_"):
            cv_top["mean_test_score"] = -cv_top["mean_test_score"]

    return {
        "pipe": rf_pipe,
        "es_clf": es_clf,
        "scoring": scoring,
        "tabla": tabla,
        "y_test": y_test,
        "y_pred": y_pred,
        "met_clf": metricas_clf(rf_pipe, X_test, y_test, y_pred) if es_clf else None,
        "imp_impureza": imp_impureza,
        "imp_perm": imp_perm,
        "best_params": busqueda.best_params_ if busqueda is not None else None,
        "cv_top": cv_top,
        "avisos": avisos,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_splits": n_splits,
        "temporal": cfg["temporal"],
        "features": list(X.columns),
    }


def empaquetar(res: dict) -> dict:
    """Convierte el resultado a algo serializable en JSON (sin el pipeline)."""
    claves = ("es_clf", "scoring", "tabla", "met_clf", "best_params", "avisos",
              "n_train", "n_test", "n_splits", "temporal", "features")
    out = {k: res[k] for k in claves}
    out["imp_impureza"] = res["imp_impureza"].head(20).to_dict()
    out["imp_perm"] = (None if res["imp_perm"] is None else
                       res["imp_perm"].head(20).reset_index()
                       .rename(columns={"index": "variable"}).to_dict("records"))
    out["cv_top"] = None if res["cv_top"] is None else res["cv_top"].to_dict("records")

    y_test, y_pred = res["y_test"], res["y_pred"]
    if res["es_clf"]:
        etiquetas = sorted(pd.Series(y_test).unique().tolist())
        out["etiquetas"] = etiquetas
        out["cm"] = confusion_matrix(y_test, y_pred, labels=etiquetas).tolist()
        out["reporte"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    else:
        out["reg"] = {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }
        out["real"] = np.asarray(y_test).tolist()
        out["pred"] = np.asarray(y_pred).tolist()
    return out
