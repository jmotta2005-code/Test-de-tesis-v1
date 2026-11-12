"""Índices espectrales derivados a partir de bandas.

Las bandas se detectan por nombre (B1, B2, ..., B8A, B11, B12) sin distinguir
mayúsculas/minúsculas. Solo se calcula un índice si el DataFrame contiene todas
las bandas que necesita.
"""
import numpy as np
import pandas as pd

BANDAS = {f"B{i}" for i in range(1, 13)} | {"B8A"}


def _dividir(a, b):
    """División que devuelve NaN cuando el denominador es 0."""
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _raiz(s):
    """Raíz cuadrada que devuelve NaN para valores negativos."""
    s = pd.Series(s, dtype="float64")
    return s.where(s >= 0).pow(0.5)


FORMULAS = {
    "EQSI 1": (("B2", "B4"), lambda b: _dividir(b["B2"], b["B4"])),
    "SI 2": (("B3", "B4"), lambda b: _dividir(b["B3"] - b["B4"], b["B3"] + b["B4"])),
    "SI 3": (("B3", "B4", "B2"), lambda b: _dividir(b["B3"] * b["B4"], b["B2"])),
    "SI 4": (("B11", "B8A"), lambda b: _dividir(b["B11"], b["B8A"])),
    "SI 5": (("B2", "B4", "B3"), lambda b: _dividir(b["B2"] * b["B4"], b["B3"])),
    "SI 6": (("B4", "B8A", "B3"), lambda b: _dividir(b["B4"] * b["B8A"], b["B3"])),
    "SI 7": (("B4", "B3"), lambda b: _raiz(b["B4"] ** 2 * b["B3"] ** 2)),
    "SI 8": (("B3", "B4"), lambda b: _raiz(b["B3"] * b["B4"])),
    "SI 9": (("B3", "B4", "B8A"), lambda b: _raiz(b["B3"] ** 2 * b["B4"] ** 2 * b["B8A"] ** 2)),
    "SI 10": (("B2", "B4"), lambda b: _raiz(b["B2"] * b["B4"])),
    "SI 11": (("B11", "B12"), lambda b: _dividir(b["B11"], b["B12"])),
    "SI 12": (("B3", "B4", "B8A"), lambda b: 2 * b["B3"] - (b["B4"] + b["B8A"])),
    "SI 13": (("B3", "B4"), lambda b: (b["B3"] + b["B4"]) / 2),
    "SI 14": (("B3", "B4", "B8A"), lambda b: (b["B3"] + b["B4"] + b["B8A"]) / 2),
    "SI 15": (("B3", "B4", "B8A"), lambda b: _raiz(b["B3"] ** 2 + b["B4"] ** 2 + b["B8A"] ** 2)),
    "SBI": (("B3", "B8A"), lambda b: _raiz(b["B3"] ** 2 + b["B8A"] ** 2)),
    "NDSI 1": (("B4", "B8A"), lambda b: _dividir(b["B4"] - b["B8A"], b["B4"] + b["B8A"])),
    "Aster - SI": (("B11", "B12"), lambda b: _dividir(b["B11"] - b["B12"], b["B11"] + b["B12"])),
    "SSSI 1": (("B4", "B8A"), lambda b: b["B4"] - b["B8A"]),
    "SSSI 2": (("B4", "B8A"), lambda b: _dividir(b["B4"] * b["B8A"] - b["B8A"] ** 2, b["B4"])),
    "BI": (("B4", "B8A"), lambda b: _raiz(b["B4"] ** 2 * b["B8A"] ** 2)),
    "CRSI": (("B8", "B4", "B3", "B2"), lambda b: _raiz(
        _dividir(b["B8"] * b["B4"] - b["B3"] * b["B2"],
                 b["B8"] * b["B4"] + b["B3"] * b["B2"]))),
    "VSSI": (("B3", "B4", "B8A"), lambda b: 2 * b["B3"] - 5 * (b["B4"] + b["B8A"])),
    "ERSSI": (("B3", "B2", "B11"), lambda b: _dividir(b["B3"] ** 2, b["B2"] * b["B11"])),
}


def _detectar_bandas(df: pd.DataFrame) -> dict:
    """Devuelve {banda_normalizada: columna_original}."""
    mapa = {}
    for col in df.columns:
        clave = str(col).strip().upper().replace(" ", "")
        if clave in BANDAS and clave not in mapa:
            mapa[clave] = col
    return mapa


def agregar_indices_espectrales(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Añade al DataFrame las columnas de índices espectrales calculables.

    Devuelve (dataframe_con_indices, nombres_de_columnas_agregadas).
    """
    bandas = _detectar_bandas(df)
    if not bandas:
        return df, []

    valores = {b: pd.to_numeric(df[col], errors="coerce").astype("float64")
               for b, col in bandas.items()}
    salida = df.copy()
    agregadas = []
    for nombre, (requeridas, formula) in FORMULAS.items():
        if nombre in salida.columns:
            continue
        if not all(b in valores for b in requeridas):
            continue
        salida[nombre] = formula(valores).to_numpy()
        agregadas.append(nombre)
    return salida, agregadas
