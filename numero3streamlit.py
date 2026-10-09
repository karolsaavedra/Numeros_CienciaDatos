
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Detector de números",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Reconocimiento de números")
st.write(
    "Dibuja un número del 1 al 9 o sube una imagen "
    "para que la inteligencia artificial lo reconozca."
)


@st.cache_resource
def cargar_modelo(ruta):
    return tf.keras.models.load_model(ruta)


# Seleccionar el modelo
opcion = st.selectbox(
    "Selecciona el modelo:",
    ["CNN con 2 convoluciones", "CNN con 3 convoluciones"]
)

if opcion == "CNN con 2 convoluciones":
    ruta_modelo = "modelos/modelo_2_conv.keras"
else:
    ruta_modelo = "modelos/modelo_3_conv.keras"


# Cargar modelo
try:
    modelo = cargar_modelo(ruta_modelo)
except Exception:
    st.error(
        f"No se pudo cargar {ruta_modelo}. "
        "Ejecuta primero el cuaderno de entrenamiento "
        "y verifica que el archivo exista."
    )
    st.stop()


def preparar_imagen(imagen):
    """Convertir una imagen a un formato similar a MNIST."""

    imagen = ImageOps.grayscale(imagen)
    imagen = np.array(imagen)

    # MNIST tiene dígitos blancos sobre fondo negro.
    # Invertir imágenes que tengan fondo predominantemente blanco.
    if imagen.mean() > 127:
        imagen = 255 - imagen

    # Eliminar el fondo y localizar el número
    mascara = imagen > 30
    coordenadas = np.argwhere(mascara)

    if len(coordenadas) == 0:
        return None

    y_min, x_min = coordenadas.min(axis=0)
    y_max, x_max = coordenadas.max(axis=0)

    imagen = imagen[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Redimensionar conservando la proporción
    alto, ancho = imagen.shape
    escala = 20 / max(alto, ancho)

    nuevo_ancho = max(1, round(ancho * escala))
    nuevo_alto = max(1, round(alto * escala))

    imagen_pil = Image.fromarray(imagen)
    imagen_pil = imagen_pil.resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    # Centrar el número en una imagen de 28 x 28
    lienzo = Image.new("L", (28, 28), 0)

    posicion = (
        (28 - nuevo_ancho) // 2,
        (28 - nuevo_alto) // 2
    )

    lienzo.paste(imagen_pil, posicion)

    return np.array(lienzo, dtype=np.float32) / 255.0



# Elegir cómo ingresar el número
modo = st.radio(
    "¿Cómo quieres ingresar el número?",
    ["Dibujar número", "Subir imagen"],
    horizontal=True
)

imagen_original = None

if modo == "Dibujar número":
    st.write("Dibuja un número del 1 al 9:")

    resultado_canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="lienzo_numeros"
    )

    # Evitar errores cuando el lienzo todavía está vacío
    try:
        datos_canvas = resultado_canvas.image_data
    except (RuntimeError, AttributeError):
        datos_canvas = None

    if datos_canvas is not None:
        if np.any(datos_canvas[:, :, 3] > 0):
            imagen_original = Image.fromarray(
                datos_canvas.astype(np.uint8)
            )

else:
    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["png", "jpg", "jpeg"],
        key="imagen_subida"
    )

    if archivo is not None:
        imagen_original = Image.open(archivo).convert("RGB")
        st.image(imagen_original, width=250)

# Botón de predicción
if st.button("Reconocer número", type="primary"):

    if imagen_original is None:
        st.warning(
            "No se recibió ningún dibujo. "
            "Dibuja el número y vuelve a pulsar el botón."
        )
    else:
        entrada = preparar_imagen(imagen_original)

        if entrada is None:
            st.warning("No se detectó ningún número.")
        else:
            entrada = entrada.reshape(1, 28, 28, 1)

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

            st.write("Probabilidades por número:")

            for i, probabilidad in enumerate(predicciones):
                st.write(f"Número {i + 1}")
                st.progress(float(probabilidad))
                st.caption(f"{probabilidad * 100:.2f}%")


# Obtener el dibujo sin provocar un error si el lienzo está vacío
try:
    datos_canvas = resultado_canvas.image_data
except RuntimeError:
    datos_canvas = None

if datos_canvas is not None:
    # Comprobar si el usuario realmente dibujó algo
    if np.any(datos_canvas[:, :, 3] > 0):
        imagen_original = Image.fromarray(
            datos_canvas.astype(np.uint8)
        )


else:
    archivo = st.file_uploader(
        "Selecciona una imagen de un número",
        type=["png", "jpg", "jpeg"]
    )

    if archivo is not None:
        imagen_original = Image.open(archivo).convert("RGB")
        st.image(
            imagen_original,
            caption="Imagen cargada",
            width=250
        )


# Realizar la predicción
if st.button("Reconocer número", type="primary"):

    if imagen_original is None:
        st.warning("Primero dibuja un número o sube una imagen.")
    else:
        entrada = preparar_imagen(imagen_original)

        if entrada is None:
            st.warning("No se detectó ningún número en la imagen.")
        else:
            entrada = entrada.reshape(1, 28, 28, 1)

            predicciones = modelo.predict(
                entrada,
                verbose=0
            )[0]

            indice = int(np.argmax(predicciones))
            numero = indice + 1
            confianza = float(predicciones[indice])

            st.success(f"Número reconocido: {numero}")

            st.metric(
                "Confianza del modelo",
                f"{confianza * 100:.2f}%"
            )

            st.write("Probabilidades por número:")

            for i, probabilidad in enumerate(predicciones):
                st.write(f"Número {i + 1}")
                st.progress(float(probabilidad))
                st.caption(f"{probabilidad * 100:.2f}%")

            st.caption(
                "La confianza es la probabilidad estimada por el modelo; "
                "no garantiza que la predicción sea correcta."
            )
