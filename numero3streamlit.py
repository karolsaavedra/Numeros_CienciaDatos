
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Detector de números",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Reconocimiento de números")
st.write("Dibuja un número del 1 al 9 o sube una imagen.")


# --------------------------------------------------
# 1. Cargar el modelo seleccionado
# --------------------------------------------------

@st.cache_resource
def cargar_modelo(ruta):
    return tf.keras.models.load_model(ruta)


opcion = st.selectbox(
    "Selecciona el modelo:",
    ["CNN con 2 convoluciones", "CNN con 3 convoluciones"]
)

if opcion == "CNN con 2 convoluciones":
    ruta_modelo = "modelos/modelo_2_conv.keras"
else:
    ruta_modelo = "modelos/modelo_3_conv.keras"

if not os.path.exists(ruta_modelo):
    st.error(f"No se encontró el modelo: {ruta_modelo}")
    st.stop()

try:
    modelo = cargar_modelo(ruta_modelo)
except Exception as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()


# --------------------------------------------------
# 2. Preparar la imagen para MNIST
# --------------------------------------------------

def preparar_imagen(imagen):
    imagen = ImageOps.grayscale(imagen)
    imagen = np.array(imagen)

    # Convertir fondo blanco a negro cuando corresponda
    if imagen.mean() > 127:
        imagen = 255 - imagen

    mascara = imagen > 30
    coordenadas = np.argwhere(mascara)

    if len(coordenadas) == 0:
        return None

    y_min, x_min = coordenadas.min(axis=0)
    y_max, x_max = coordenadas.max(axis=0)

    imagen = imagen[y_min:y_max + 1, x_min:x_max + 1]

    alto, ancho = imagen.shape
    escala = 20 / max(alto, ancho)

    nuevo_ancho = max(1, round(ancho * escala))
    nuevo_alto = max(1, round(alto * escala))

    imagen_pil = Image.fromarray(imagen).resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    lienzo = Image.new("L", (28, 28), 0)

    lienzo.paste(
        imagen_pil,
        ((28 - nuevo_ancho) // 2, (28 - nuevo_alto) // 2)
    )

    return np.array(lienzo, dtype=np.float32) / 255.0


# --------------------------------------------------
# 3. Dibujar o subir una imagen
# --------------------------------------------------

modo = st.radio(
    "¿Cómo quieres ingresar el número?",
    ["Dibujar número", "Subir imagen"],
    horizontal=True
)

imagen_original = None


if modo == "Dibujar número":
    st.info("Dibuja un número del 1 al 9 dentro del cuadro negro.")

    resultado_canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="lienzo_numeros",
        update_streamlit=True
    )

    datos_canvas = None

    if resultado_canvas is not None:
        try:
            datos_canvas = resultado_canvas.image_data
        except (RuntimeError, AttributeError):
            datos_canvas = None

    resumen_canvas = "el componente no entregó image_data"
    if datos_canvas is not None:
        datos_canvas = np.asarray(datos_canvas)
        if datos_canvas.ndim == 2:
            hay_trazo = bool(np.any(datos_canvas > 30))
        elif datos_canvas.ndim == 3 and datos_canvas.shape[2] >= 3:
            hay_trazo = bool(np.any(datos_canvas[:, :, :3] > 30))
        else:
            hay_trazo = False

        resumen_canvas = f"image_data={datos_canvas.shape}, trazo={hay_trazo}"
        if hay_trazo:
            st.session_state["imagen_dibujo"] = datos_canvas.astype(np.uint8)

    datos_dibujo = st.session_state.get("imagen_dibujo")
    if datos_dibujo is not None:
        imagen_original = Image.fromarray(datos_dibujo)
        st.image(imagen_original, caption="Dibujo detectado", width=200)
    else:
        st.info("Todavía no se recibió un trazo del lienzo.")

    st.caption(f"Diagnóstico del lienzo: {resumen_canvas}")

else:
    st.session_state.pop("imagen_dibujo", None)
    archivo = st.file_uploader(
        "Selecciona una imagen PNG, JPG o JPEG",
        type=["png", "jpg", "jpeg"],
        key="imagen_subida"
    )

    if archivo is not None:
        imagen_original = Image.open(archivo).convert("RGB")
        st.image(imagen_original, caption="Imagen seleccionada", width=250)



# --------------------------------------------------
# 4. Realizar una sola predicción
# --------------------------------------------------

if st.button("Reconocer número", type="primary"):

    if imagen_original is None:
        st.warning(
            "No se recibió ninguna imagen. "
            "Dibuja un número o selecciona un archivo."
        )

    else:
        entrada = preparar_imagen(imagen_original)

        if entrada is None:
            st.warning("No se detectó ningún número en la imagen.")

        else:
            entrada = entrada.reshape(1, 28, 28, 1)

            try:
                predicciones = modelo.predict(
                    entrada,
                    verbose=0
                )[0]

                indice = int(np.argmax(predicciones))
                numero = indice + 1
                confianza = float(predicciones[indice])

                st.success(f"Número reconocido: {numero}")
                st.metric(
                    "Confianza",
                    f"{confianza * 100:.2f}%"
                )

                st.subheader("Probabilidades por número")

                for i, probabilidad in enumerate(predicciones):
                    st.write(f"Número {i + 1}")
                    st.progress(float(probabilidad))

            except Exception as error:
                st.error(f"Error al realizar la predicción: {error}")
