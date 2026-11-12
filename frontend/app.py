"""Interfaz Streamlit. Solo muestra y pide datos; todo el ML vive en el backend.

Ejecutar (desde la carpeta frontend):
    streamlit run app.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import api

st.set_page_config(page_title="Random Forest Trainer", page_icon="🌲", layout="wide")

LBL = {"base": "Línea base (trivial)", "simple": "Modelo simple (logística / lineal)",
       "rf": "Random Forest"}


def val(scoring, v):
    return -v if scoring.startswith("neg_") else v


@st.cache_data(show_spinner=False)
def c_hojas(nombre, tam, _contenido):
    return api.hojas(nombre, _contenido)


@st.cache_data(show_spinner=False)
def c_preparar(did, cfg_json):
    return api.preparar(did, json.loads(cfg_json))


def elegir_hoja(archivo, clave):
    nombres = c_hojas(archivo.name, archivo.size, archivo.getvalue())
    if len(nombres) > 1:
        return st.selectbox(f"Hoja de '{archivo.name}'", nombres, key=f"hoja_{clave}")
    return None


def fig_barh(serie, err=None, color="#2e7d32", xlabel="Importancia"):
    serie = serie[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(serie))))
    ax.barh([str(i) for i in serie.index], serie.values,
            xerr=None if err is None else err[::-1].values, color=color)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- backend
try:
    conf = api.config()
except Exception:  # noqa: BLE001
    st.error(f"No se pudo conectar con el backend en {api.API_URL}. "
             "¿Está corriendo `uvicorn main:app --port 8000`?")
    st.stop()

# ------------------------------------------------------------ barra lateral
st.sidebar.header("⚙️ Configuración")
st.sidebar.subheader("Random Forest (sin búsqueda)")
n_estimators = st.sidebar.slider("n_estimators", 10, 1000, 200, 10)
max_depth_on = st.sidebar.checkbox("Limitar profundidad máxima", value=False)
max_depth = st.sidebar.slider("max_depth", 2, 50, 10) if max_depth_on else None
min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 2)
min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 50, 1)
_mf = {"sqrt": "sqrt", "log2": "log2", "0.5": 0.5, "todas (None)": None}
max_features = _mf[st.sidebar.selectbox("max_features", list(_mf), index=0)]

st.sidebar.subheader("Partición y validación")
test_size = st.sidebar.slider("Tamaño del conjunto de prueba", 0.05, 0.5, 0.2, 0.05)
n_splits = st.sidebar.slider("Folds de validación cruzada", 2, 10, 5)
random_state = int(st.sidebar.number_input("random_state", 0, 10_000, 42))

st.sidebar.subheader("Variables categóricas")
umbral_card = st.sidebar.slider("Alta cardinalidad: más de N categorías", 5, 200, 30)
modo_card_ui = st.sidebar.radio("Tratamiento de alta cardinalidad",
                                ["Agrupar categorías raras en 'otros'", "Excluir esas columnas"])
modo_card = "excluir" if modo_card_ui.startswith("Excluir") else "agrupar"
top_n = st.sidebar.slider("Categorías a conservar (el resto → 'otros')", 5, 100, 20)

# ---------------------------------------------------------- 1. carga
st.title("🌲 Panel de entrenamiento – Random Forest")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("1️⃣ Variables de entrenamiento (X)")
    archivo_x = st.file_uploader("CSV o Excel con las features", type=["csv", "xlsx", "xls"], key="file_x")
    hoja_x = elegir_hoja(archivo_x, "x") if archivo_x else None
with col_b:
    st.subheader("2️⃣ Variable objetivo (y)")
    archivo_y = st.file_uploader("CSV o Excel con el target", type=["csv", "xlsx", "xls"], key="file_y")
    hoja_y = elegir_hoja(archivo_y, "y") if archivo_y else None

if not (archivo_x and archivo_y):
    st.info("Sube ambos archivos para continuar.")
    st.stop()

clave = (archivo_x.name, archivo_x.size, hoja_x, archivo_y.name, archivo_y.size, hoja_y)
if st.session_state.get("ds_clave") != clave:
    try:
        with st.spinner("Enviando archivos al backend..."):
            st.session_state["ds"] = api.crear_dataset(archivo_x, archivo_y, hoja_x, hoja_y)
        st.session_state["ds_clave"] = clave
        st.session_state.pop("resultado", None)
        st.session_state.pop("fuga", None)
    except Exception as e:  # noqa: BLE001
        st.error(f"{e}")
        st.stop()

ds = st.session_state["ds"]
did = ds["dataset_id"]

with st.expander("Vista previa de los datos"):
    p1, p2 = st.columns(2)
    p1.write(f"**Features** – {ds['filas_x']} filas × {len(ds['columnas_x'])} columnas")
    p1.dataframe(pd.DataFrame(ds["preview_x"]))
    p2.write(f"**Target** – {ds['filas_y']} filas × {len(ds['columnas_y'])} columnas")
    p2.dataframe(pd.DataFrame(ds["preview_y"]))

# ---------------------------------------------------------- 2. preparación
st.subheader("3️⃣ Preparación")
c1, c2 = st.columns(2)
with c1:
    if len(ds["columnas_y"]) == 1:
        col_objetivo = ds["columnas_y"][0]
        st.write(f"Columna objetivo detectada: **{col_objetivo}**")
    else:
        col_objetivo = st.selectbox("Columna objetivo en el archivo de target", ds["columnas_y"])
with c2:
    comunes = [c for c in ds["columnas_x"] if c in ds["columnas_y"] and c != col_objetivo]
    opciones_union = ["Por posición (mismo orden de filas)"]
    if comunes:
        opciones_union.append("Por columna ID común")
    modo_union = st.radio("¿Cómo emparejar las filas de ambos archivos?", opciones_union)

col_id = st.selectbox("Columna ID", comunes) if modo_union.startswith("Por columna ID") else None

st.markdown("**Orden temporal**")
temporal = st.checkbox(
    "Mis datos tienen orden temporal (predecir el futuro con el pasado)",
    help="El conjunto de prueba serán las filas MÁS RECIENTES y la validación cruzada "
         "respetará el orden cronológico.")
col_tiempo, excluir_tiempo = None, True
if temporal:
    t1, t2 = st.columns(2)
    col_tiempo = t1.selectbox("Columna de fecha / orden",
                              [c for c in ds["columnas_x"] if c != col_objetivo])
    excluir_tiempo = t2.checkbox("Excluir esa columna de las variables", value=True)

candidatas = [c for c in ds["columnas_x"]
              if c not in (col_objetivo, col_id) and not (temporal and excluir_tiempo and c == col_tiempo)]
excluir = st.multiselect("Columnas a excluir del entrenamiento (opcional)", candidatas)
auto_excl = st.checkbox("Excluir automáticamente columnas constantes, de fecha o que parecen IDs", value=True)
quitar_dup = st.checkbox("Eliminar filas duplicadas (recomendado)", value=True)

cfg_prep = dict(col_objetivo=col_objetivo, col_id=col_id, temporal=temporal, col_tiempo=col_tiempo,
                excluir_tiempo=excluir_tiempo, excluir=excluir, auto_excl=auto_excl,
                umbral_card=umbral_card, modo_card=modo_card, quitar_dup=quitar_dup)
try:
    info = c_preparar(did, json.dumps(cfg_prep, sort_keys=True))
except Exception as e:  # noqa: BLE001
    st.error(f"{e}")
    st.stop()

if col_id:
    st.caption(f"Filas tras la unión: {info['filas_union']}")
if info["filas_sin_target"]:
    st.caption(f"Se descartaron {info['filas_sin_target']} filas sin valor en el objetivo.")
if info["motivos"]:
    st.info("Excluidas automáticamente: " + "; ".join(f"**{c}** ({m})" for c, m in info["motivos"].items()))
if info["cat_alta"]:
    if modo_card == "excluir":
        st.info(f"Columnas de alta cardinalidad excluidas: {', '.join(info['cat_alta'])}")
    else:
        st.info(f"Columnas de alta cardinalidad (se conservan las {top_n} categorías más frecuentes "
                f"y el resto va a 'otros'): {', '.join(info['cat_alta'])}")
if info["n_dup"] > 0:
    st.warning(f"Había **{info['n_dup']}** filas duplicadas. Si quedan repartidas entre train y test, "
               "inflan las métricas.")

tipo = st.radio("Tipo de problema", ["Clasificación", "Regresión"],
                index=0 if info["sugerido_clf"] else 1, horizontal=True)
es_clf = tipo == "Clasificación"

if info["n_filas"] < 30:
    st.error("Hay muy pocas filas (menos de 30) para entrenar y validar de forma fiable.")
    st.stop()
st.write(f"**{info['n_filas']}** filas · **{info['n_vars']}** variables · "
         + (f"**{info['n_clases']}** clases" if es_clf else "target continuo"))

# ---------------------------------------------------------- 3. diagnóstico
st.subheader("4️⃣ Diagnóstico de datos")
tab_cal, tab_obj, tab_fuga = st.tabs(["🧹 Calidad de datos", "🎯 Variable objetivo",
                                      "🕵️ Posible fuga de información"])
with tab_cal:
    m1, m2, m3 = st.columns(3)
    m1.metric("Filas", info["n_filas"])
    m2.metric("Variables", info["n_vars"])
    m3.metric("% de celdas faltantes", f"{info['pct_faltantes']:.1f}%")
    st.dataframe(pd.DataFrame(info["diagnostico"]))

desbalanceado = False
with tab_obj:
    if es_clf:
        if info["distribucion"] is None:
            st.warning("El objetivo tiene demasiados valores distintos para ser clasificación.")
        else:
            st.bar_chart(pd.Series(info["distribucion"]))
            desbalanceado = info["desbalanceado"]
            if desbalanceado:
                st.warning(f"Clases desbalanceadas: la mayoritaria es el {info['prop_mayor']:.0%} de los datos. "
                           "Usa F1 / balanced accuracy / ROC AUC y `class_weight`.")
            else:
                st.success("Las clases están razonablemente balanceadas.")
            if info["clase_min"]["n"] < 10:
                st.warning(f"Alguna clase tiene menos de 10 muestras "
                           f"({info['clase_min']['clase']}: {info['clase_min']['n']}).")
    else:
        if info["hist"] is None:
            st.error("Para regresión el target debe ser numérico.")
        else:
            e = np.array(info["hist"]["edges"])
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(e[:-1], info["hist"]["counts"], width=np.diff(e), align="edge", color="#2e7d32")
            ax.set_xlabel(col_objetivo)
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame(info["describe"]))

with tab_fuga:
    st.write("Se entrena un árbol pequeño con **una sola variable a la vez**. Si una variable por sí "
             "sola predice casi perfecto, probablemente contiene la respuesta.")
    umbral_fuga = st.slider("Umbral de sospecha", 0.5, 1.0, 0.9, 0.01)
    if st.button("🔍 Escanear variables"):
        try:
            with st.spinner("Escaneando variables..."):
                st.session_state["fuga"] = api.fuga(did, dict(cfg_prep, es_clf=es_clf,
                                                              random_state=random_state))
        except Exception as e:  # noqa: BLE001
            st.error(f"{e}")
    fg = st.session_state.get("fuga")
    if fg:
        df_fuga = pd.DataFrame(fg["tabla"])
        susp = df_fuga[df_fuga["score con 1 sola variable"] > umbral_fuga]
        if len(susp):
            st.error(f"{len(susp)} variable(s) predicen el objetivo con {fg['metrica']} > {umbral_fuga:.2f} "
                     "usando solo ellas: " + ", ".join(susp["columna"].astype(str)) +
                     ". Revisa si son legítimas; si no, exclúyelas arriba.")
        else:
            st.success("Ninguna variable individual supera el umbral. (No descarta fugas combinadas.)")
        st.dataframe(df_fuga.head(30))

# ---------------------------------------------------------- 4. entrenamiento
st.subheader("5️⃣ Entrenamiento")
k1, k2 = st.columns(2)
with k1:
    if es_clf:
        opciones = dict(conf["clf_scoring"])
        opciones["ROC AUC"] = "roc_auc" if info["n_clases"] == 2 else "roc_auc_ovr"
    else:
        opciones = dict(conf["reg_scoring"])
    nombre_score = st.selectbox("Métrica principal (CV, búsqueda y comparación)", list(opciones))
    scoring = opciones[nombre_score]
    class_weight = None
    if es_clf:
        cw_ui = st.selectbox("class_weight", ["balanced", "balanced_subsample", "ninguno"],
                             index=0 if desbalanceado else 2)
        class_weight = None if cw_ui == "ninguno" else cw_ui
with k2:
    search_mode = st.radio("Búsqueda de hiperparámetros", ["Ninguna", "Aleatoria", "Rejilla"],
                           horizontal=True,
                           help="Usa solo el train. Si está activa, la barra lateral de hiperparámetros se ignora.")
    n_iter = 20
    if search_mode == "Aleatoria":
        n_iter = st.slider("Combinaciones a probar", 5, 100, 20)
        st.caption(f"≈ {n_iter * n_splits} entrenamientos.")
    elif search_mode == "Rejilla":
        st.caption(f"{conf['combos_rejilla']} combinaciones × {n_splits} folds = "
                   f"{conf['combos_rejilla'] * n_splits} entrenamientos.")
    usar_perm = st.checkbox("Calcular importancia por permutación (más lento, más fiable)", value=True)

if st.button("🚀 Entrenar y evaluar", type="primary"):
    cfg_train = dict(
        cfg_prep, es_clf=es_clf, random_state=random_state, test_size=test_size, n_splits=n_splits,
        scoring=scoring, class_weight=class_weight, search_mode=search_mode, n_iter=n_iter,
        usar_perm=usar_perm, top_n=top_n,
        rf_params=dict(n_estimators=n_estimators, max_depth=max_depth,
                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                       max_features=max_features),
    )
    try:
        with st.spinner("Entrenando y validando modelos (puede tardar)..."):
            r = api.entrenar(did, cfg_train)
            r["nombre_score"] = nombre_score
            r["modelo_bytes"] = api.descargar_modelo(r["modelo_id"])
        st.session_state["resultado"] = r
    except Exception as e:  # noqa: BLE001
        st.error(f"{e}")

# ---------------------------------------------------------- 5. resultados
res = st.session_state.get("resultado")
if res:
    sc, nombre_sc, tabla = res["scoring"], res["nombre_score"], res["tabla"]
    st.header("📊 Resultados")
    st.caption(f"Train: {res['n_train']} filas · Test: {res['n_test']} filas · {res['n_splits']} folds · "
               "partición " + ("temporal (test = lo más reciente)" if res["temporal"] else "aleatoria"))
    for a in res["avisos"]:
        st.warning(a)

    st.subheader(f"Comparación de modelos ({nombre_sc})")
    st.dataframe(pd.DataFrame([{
        "Modelo": LBL[k],
        "CV (media ± desv.)": f"{val(sc, t['cv_mean']):.3f} ± {t['cv_std']:.3f}",
        "Train (CV)": f"{val(sc, t['train_mean']):.3f}",
        "Test": f"{val(sc, t['test']):.3f}",
    } for k, t in tabla.items()]))
    if sc.startswith("neg_"):
        st.caption("Para RMSE y MAE, menor es mejor.")

    rf = tabla["rf"]
    mejor_ref = max(tabla["base"]["cv_mean"], tabla["simple"]["cv_mean"])
    if rf["cv_mean"] - mejor_ref <= rf["cv_std"]:
        st.warning("El Random Forest **no supera claramente** a la línea base o al modelo simple "
                   "(la mejora en CV es menor que la variabilidad entre folds).")
    else:
        st.success(f"El Random Forest mejora a la mejor referencia en CV "
                   f"({val(sc, rf['cv_mean']):.3f} vs {val(sc, mejor_ref):.3f}).")

    st.subheader("Revisión de sobreajuste")
    brecha_rel = (rf["train_mean"] - rf["cv_mean"]) / (abs(rf["cv_mean"]) + 1e-9)
    b1, b2, b3 = st.columns(3)
    b1.metric("Train (CV)", f"{val(sc, rf['train_mean']):.3f}")
    b2.metric("Validación (CV)", f"{val(sc, rf['cv_mean']):.3f}")
    b3.metric("Test", f"{val(sc, rf['test']):.3f}")
    if brecha_rel > 0.15:
        st.warning(f"Brecha train–validación de {brecha_rel:.0%}. Prueba `max_depth` menor o "
                   "`min_samples_leaf` mayor; lo importante es que CV y test estén cerca.")
    else:
        st.success("Brecha train–validación pequeña.")
    if abs(rf["test"] - rf["cv_mean"]) > max(2 * rf["cv_std"], 0.05 * abs(rf["cv_mean"])):
        st.warning("El resultado en test difiere bastante del de validación cruzada "
                   "(test pequeño, drift o mala suerte en la partición).")
    if not sc.startswith("neg_") and rf["test"] > 0.98:
        st.error("Resultado sospechosamente alto. Revisa fuga de información y duplicados.")

    if res["best_params"]:
        st.subheader("Mejores hiperparámetros encontrados")
        st.json({k.replace("rf__", ""): v for k, v in res["best_params"].items()})
        st.write("Top 10 combinaciones (validación cruzada en train):")
        st.dataframe(pd.DataFrame(res["cv_top"]))
        st.caption("El score de CV es algo optimista; el de Test es la referencia más honesta.")

    st.subheader("Detalle del Random Forest en el conjunto de prueba")
    if res["es_clf"]:
        met = res["met_clf"]
        for col, (k, v) in zip(st.columns(len(met)), met.items()):
            col.metric(k, "n/d" if v is None else f"{v:.3f}")
        g1, g2 = st.columns(2)
        with g1:
            st.write("**Matriz de confusión**")
            cm, et = np.array(res["cm"]), [str(x) for x in res["etiquetas"]]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(et)))
            ax.set_xticklabels(et, rotation=45, ha="right")
            ax.set_yticks(range(len(et)))
            ax.set_yticklabels(et)
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            for i in range(len(et)):
                for j in range(len(et)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            st.write("**Reporte por clase** (precision, recall y F1)")
            st.dataframe(pd.DataFrame(res["reporte"]).T.round(3))
    else:
        r_ = res["reg"]
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{r_['r2']:.3f}")
        m2.metric("RMSE", f"{r_['rmse']:.3f}")
        m3.metric("MAE", f"{r_['mae']:.3f}")
        st.write("**Real vs. predicho**")
        real, pred = np.array(res["real"]), np.array(res["pred"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(real, pred, alpha=0.6)
        lim = [min(real.min(), pred.min()), max(real.max(), pred.max())]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Real")
        ax.set_ylabel("Predicho")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Importancia de variables")
    i1, i2 = st.columns(2)
    with i1:
        st.write("**Por impureza** (top 20, sobre columnas transformadas)")
        fig = fig_barh(pd.Series(res["imp_impureza"]))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Puede favorecer variables con muchos valores distintos.")
    with i2:
        if res["imp_perm"]:
            st.write(f"**Por permutación** (top 20, caída en {nombre_sc} en test)")
            top = pd.DataFrame(res["imp_perm"]).set_index("variable")
            fig = fig_barh(top["media"], err=top["std"], color="#1565c0", xlabel="Caída del score")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Más fiable. Valores cercanos a 0 o negativos: la variable no aporta.")
        else:
            st.info("Activa la importancia por permutación para ver esta gráfica.")

    st.download_button("💾 Descargar modelo (.joblib)", data=res["modelo_bytes"],
                       file_name="random_forest_model.joblib", mime="application/octet-stream")

    st.subheader("6️⃣ Predecir con datos nuevos (opcional)")
    st.caption("Usa datos con distribución parecida a la de entrenamiento; si cambian con el tiempo, reentrena.")
    nuevo = st.file_uploader("CSV o Excel con las mismas variables de entrenamiento",
                             type=["csv", "xlsx", "xls"], key="file_nuevo")
    if nuevo:
        try:
            out = api.predecir(res["modelo_id"], nuevo.name, nuevo.getvalue())
            st.dataframe(pd.DataFrame(out["preview"]))
            st.download_button("⬇️ Descargar predicciones (CSV)", data=out["csv"].encode("utf-8"),
                               file_name="predicciones.csv", mime="text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"{e}")
