
# Reconocimiento de Números con Redes Neuronales Convolucionales (CNN)

## Descripción del proyecto

Este proyecto consiste en desarrollar una aplicación web capaz de reconocer números escritos a mano mediante redes neuronales convolucionales (CNN). Se entrenaron dos modelos de aprendizaje profundo: uno con dos capas convolucionales y otro con tres, con el propósito de comparar su funcionamiento en la clasificación de imágenes.

La aplicación permite dibujar un número directamente en la página web o cargar una imagen desde el computador. Posteriormente, el modelo seleccionado analiza la imagen y predice qué número representa, mostrando el resultado y su nivel de confianza.

## Tecnologías utilizadas

- **Python:** lenguaje de programación utilizado para desarrollar el proyecto.
- **TensorFlow y Keras:** construcción, entrenamiento y ejecución de los modelos.
- **NumPy:** procesamiento de matrices y datos numéricos.
- **Pillow (PIL):** procesamiento y transformación de imágenes.
- **Streamlit:** creación de la interfaz web interactiva.
- **Streamlit Drawable Canvas:** herramienta para dibujar números en la aplicación.

## Modelos implementados

### 1. CNN con dos capas convolucionales

Este modelo utiliza dos capas convolucionales para extraer características de las imágenes, capas de agrupamiento (*MaxPooling*) para reducir sus dimensiones y capas densas para realizar la clasificación.

**Archivo:** `modelos/modelo_2_conv.keras`

### 2. CNN con tres capas convolucionales

Este modelo incorpora tres capas convolucionales para aprender características visuales adicionales. Posteriormente, utiliza capas densas para clasificar la imagen recibida.

**Archivo:** `modelos/modelo_3_conv.keras`

Ambos modelos reciben imágenes de 28 × 28 píxeles y clasifican los números del 1 al 9, distribuidos en nueve clases. El número cero no se incluye entre las clases de salida de la aplicación.

## ¿Cómo funciona la aplicación?

1. **Selección del modelo:** el usuario elige entre la CNN de dos o tres capas convolucionales.
2. **Entrada de la imagen:** puede dibujar un número directamente en el lienzo o cargar una imagen en formato PNG, JPG o JPEG.
3. **Preprocesamiento:** la imagen se convierte a escala de grises y se identifica el área donde está el número.
4. **Ajuste de dimensiones:** se recorta el espacio vacío y se centra el número en una imagen de 28 × 28 píxeles.
5. **Normalización:** los valores de los píxeles se convierten a un rango entre 0 y 1.
6. **Predicción:** la imagen se procesa mediante el modelo seleccionado.
7. **Visualización:** se muestra el número reconocido, la confianza estimada y las probabilidades de las nueve clases.

## Aplicación web

**Enlace de acceso:** [Detección de Números](https://numeroscienciadatos-yeua6yrsg6jdbqpkjjszon.streamlit.app/)

La aplicación permite seleccionar un modelo, dibujar o cargar una imagen y consultar la predicción obtenida.

## Estructura del repositorio

```text
Numeros_CienciaDatos/
├── 8 A2_CNN.ipynb
├── numero3streamlit.py
├── requirements.txt
├── modelos/
│   ├── modelo_2_conv.keras
│   └── modelo_3_conv.keras
└── README.md
```

| Archivo o carpeta | Descripción |
|---|---|
| `8 A2_CNN.ipynb` | Notebook con el desarrollo y entrenamiento de los modelos. |
| `numero3streamlit.py` | Código fuente de la aplicación web. |
| `requirements.txt` | Dependencias necesarias para ejecutar el proyecto. |
| `modelos/` | Carpeta con los modelos entrenados. |
| `README.md` | Documentación del proyecto. |

##  Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/karolsaavedra/Numeros_CienciaDatos.git
cd Numeros_CienciaDatos
```

### 2. Crear un entorno virtual

En Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run numero3streamlit.py
```
- La precisión puede variar según la calidad de la imagen, la forma del número y los datos utilizados durante el entrenamiento.
- La aplicación está diseñada para reconocer números del 1 al 9.

## Estudiante

**Karol Stefany Saavedra Suárez**

Proyecto académico de Ciencia de Datos.
