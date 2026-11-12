"""API del entrenador de Random Forest.

Ejecutar (desde la carpeta backend):
    uvicorn main:app --reload --port 8000
Documentación interactiva: http://127.0.0.1:8000/docs
"""
import io
import json
import math
import uuid
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import indices
import ml_core as ml

app = FastAPI(title="Random Forest Trainer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Estado en memoria (suficiente para uso personal; se pierde al reiniciar).
DATASETS: dict = {}
MODELOS: dict = {}


# ------------------------------------------------------------------ helpers
def limpio(o):
    """Hace serializable cualquier estructura (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [limpio(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if o is None or isinstance(o, (str, bool, int)):
        return o
    return str(o)


def a_json(df: pd.DataFrame):
    return json.loads(df.to_json(orient="records"))


# ------------------------------------------------------------------ esquemas
class PrepConfig(BaseModel):
    col_objetivo: str
    col_id: Optional[str] = None
    temporal: bool = False
    col_tiempo: Optional[str] = None
    excluir_tiempo: bool = True
    excluir: list = []
    auto_excl: bool = True
    umbral_card: int = 30
    modo_card: str = "agrupar"  # "agrupar" | "excluir"
    quitar_dup: bool = True


class FugaReq(PrepConfig):
    es_clf: bool
    random_state: int = 42


class TrainConfig(PrepConfig):
    es_clf: bool
    random_state: int = 42
    test_size: float = 0.2
    n_splits: int = 5
    scoring: str
    class_weight: Optional[str] = None
    search_mode: str = "Ninguna"  # "Ninguna" | "Aleatoria" | "Rejilla"
    n_iter: int = 20
    usar_perm: bool = True
    top_n: int = 20
    rf_params: dict


def _datos(did: str, cfg: PrepConfig):
    ds = DATASETS.get(did)
    if ds is None:
        raise HTTPException(404, "Dataset no encontrado. Vuelve a subir los archivos.")
    try:
        return ml.preparar_datos(ds["x"], ds["y"], cfg.model_dump())
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ----------------------------------------------------------------- endpoints
@app.get("/config")
def config():
    combos = int(np.prod([len(v) for v in ml.ESPACIO_REJILLA.values()]))
    return {"clf_scoring": ml.CLF_SCORING, "reg_scoring": ml.REG_SCORING, "combos_rejilla": combos}


@app.post("/hojas")
def hojas(archivo: UploadFile = File(...)):
    try:
        return {"hojas": ml.hojas_excel(archivo.file.read(), archivo.filename)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo leer '{archivo.filename}': {e}")


@app.post("/datasets")
def crear_dataset(
    archivo_x: UploadFile = File(...),
    archivo_y: UploadFile = File(...),
    hoja_x: Optional[str] = Form(None),
    hoja_y: Optional[str] = Form(None),
):
    try:
        df_x = ml.leer_archivo(archivo_x.file.read(), archivo_x.filename, hoja_x)
        df_y = ml.leer_archivo(archivo_y.file.read(), archivo_y.filename, hoja_y)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    df_x, idx_x = indices.agregar_indices_espectrales(df_x)
    df_y, idx_y = indices.agregar_indices_espectrales(df_y)
    did = uuid.uuid4().hex
    DATASETS[did] = {"x": df_x, "y": df_y}
    return {
        "dataset_id": did,
        "filas_x": len(df_x), "filas_y": len(df_y),
        "columnas_x": list(map(str, df_x.columns)),
        "columnas_y": list(map(str, df_y.columns)),
        "indices_agregados_x": idx_x,
        "indices_agregados_y": idx_y,
        "preview_x": a_json(df_x.head(10)),
        "preview_y": a_json(df_y.head(10)),
    }


@app.post("/datasets/{did}/preparar")
def preparar(did: str, cfg: PrepConfig):
    X, y, info = _datos(did, cfg)
    resp = dict(info)
    resp.update(
        n_filas=len(X), n_vars=int(X.shape[1]), features=list(X.columns),
        pct_faltantes=float(100 * X.isna().to_numpy().mean()),
        diagnostico=a_json(ml.diagnostico_columnas(X, cfg.umbral_card)),
        sugerido_clf=ml.es_clasificacion(y),
        n_clases=int(y.nunique()),
        distribucion=None, desbalanceado=False, prop_mayor=None, clase_min=None,
        hist=None, describe=None,
    )
    if y.nunique() <= 50:
        dist = y.value_counts()
        prop = dist.max() / len(y)
        resp["distribucion"] = {str(k): int(v) for k, v in dist.items()}
        resp["prop_mayor"] = float(prop)
        resp["desbalanceado"] = bool(prop > 0.7 or dist.max() / max(dist.min(), 1) > 5)
        resp["clase_min"] = {"clase": str(dist.idxmin()), "n": int(dist.min())}
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        c, e = np.histogram(y.dropna(), bins=30)
        resp["hist"] = {"counts": c.tolist(), "edges": e.tolist()}
        resp["describe"] = a_json(y.describe().to_frame().T)
    return limpio(resp)


@app.post("/datasets/{did}/fuga")
def fuga(did: str, cfg: FugaReq):
    X, y, _ = _datos(did, cfg)
    df, metrica = ml.escanear_fuga(X, y, cfg.es_clf, cfg.random_state)
    return limpio({"tabla": a_json(df), "metrica": metrica})


@app.post("/datasets/{did}/entrenar")
def entrenar(did: str, cfg: TrainConfig):
    X, y, _ = _datos(did, cfg)
    if not cfg.es_clf and not pd.api.types.is_numeric_dtype(y):
        raise HTTPException(400, "Para regresión el target debe ser numérico.")
    if len(X) < 30:
        raise HTTPException(400, "Hay muy pocas filas (menos de 30) para entrenar y validar.")
    try:
        res = ml.entrenar_y_evaluar(X, y, cfg.es_clf, cfg.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Error durante el entrenamiento: {e}")
    mid = uuid.uuid4().hex
    MODELOS[mid] = {"pipeline": res["pipe"], "features": res["features"], "target": cfg.col_objetivo}
    out = ml.empaquetar(res)
    out.update(modelo_id=mid, target=cfg.col_objetivo)
    return limpio(out)


@app.get("/modelos/{mid}/descargar")
def descargar(mid: str):
    m = MODELOS.get(mid)
    if m is None:
        raise HTTPException(404, "Modelo no encontrado.")
    buf = io.BytesIO()
    joblib.dump({"pipeline": m["pipeline"], "features": m["features"], "target": m["target"]}, buf)
    return Response(buf.getvalue(), media_type="application/octet-stream")


@app.post("/modelos/{mid}/predecir")
def predecir(mid: str, archivo: UploadFile = File(...)):
    m = MODELOS.get(mid)
    if m is None:
        raise HTTPException(404, "Modelo no encontrado.")
    try:
        df = ml.leer_archivo(archivo.file.read(), archivo.filename)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")
    df, _ = indices.agregar_indices_espectrales(df)
    faltan = [c for c in m["features"] if c not in df.columns]
    if faltan:
        raise HTTPException(400, f"Faltan columnas en el archivo nuevo: {faltan}")
    try:
        preds = m["pipeline"].predict(ml.preparar_X(df[m["features"]]))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"No se pudo predecir: {e}")
    salida = df.copy()
    salida[f"prediccion_{m['target']}"] = preds
    return limpio({"preview": a_json(salida.head(50)), "csv": salida.to_csv(index=False)})
