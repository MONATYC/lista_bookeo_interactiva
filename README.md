# Lista Bookeo Interactiva

Esta aplicación permite extraer y gestionar visitas de archivos PDF exportados desde Bookeo. Utiliza Streamlit y la API de Google Generative AI para procesar el documento y mostrar los datos de forma interactiva.

## Instalación

1. Clona este repositorio.
2. Instala las dependencias de Python:

```bash
pip install -r requirements.txt
```

## Configuración de secretos

Crea un archivo `secrets.toml` en la raíz del proyecto con tu clave de API:

```toml
[api]
API_KEY = "TU_CLAVE_AQUI"
```

## Uso

Ejecuta la aplicación con Streamlit:

```bash
streamlit run app.py
```

Al abrir la URL indicada, podrás subir un PDF de Bookeo y gestionar las visitas de forma interactiva.
