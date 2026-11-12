"""Cliente HTTP del backend."""
import os

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    pass


def _ok(r):
    if not r.ok:
        try:
            detalle = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detalle = r.text
        raise ApiError(detalle)
    return r


def config():
    return _ok(requests.get(f"{API_URL}/config", timeout=15)).json()


def hojas(nombre, contenido):
    r = requests.post(f"{API_URL}/hojas", files={"archivo": (nombre, contenido)}, timeout=60)
    return _ok(r).json()["hojas"]


def crear_dataset(fx, fy, hoja_x=None, hoja_y=None):
    files = {"archivo_x": (fx.name, fx.getvalue()), "archivo_y": (fy.name, fy.getvalue())}
    data = {}
    if hoja_x:
        data["hoja_x"] = hoja_x
    if hoja_y:
        data["hoja_y"] = hoja_y
    return _ok(requests.post(f"{API_URL}/datasets", files=files, data=data, timeout=300)).json()


def preparar(did, cfg):
    return _ok(requests.post(f"{API_URL}/datasets/{did}/preparar", json=cfg, timeout=120)).json()


def fuga(did, cfg):
    return _ok(requests.post(f"{API_URL}/datasets/{did}/fuga", json=cfg, timeout=600)).json()


def entrenar(did, cfg):
    return _ok(requests.post(f"{API_URL}/datasets/{did}/entrenar", json=cfg, timeout=None)).json()


def descargar_modelo(mid):
    return _ok(requests.get(f"{API_URL}/modelos/{mid}/descargar", timeout=120)).content


def predecir(mid, nombre, contenido):
    r = requests.post(f"{API_URL}/modelos/{mid}/predecir",
                      files={"archivo": (nombre, contenido)}, timeout=300)
    return _ok(r).json()
