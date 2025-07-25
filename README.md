# Lista Bookeo Interactiva

Esta aplicación permite extraer y gestionar visitas de archivos PDF exportados desde Bookeo. Utiliza Streamlit y la API de Google Generative AI para procesar el documento y mostrar los datos de forma interactiva.

> **Nota técnica**:  
> Las primeras líneas de `app.py` establecen la variable de entorno `WATCHDOG_USE_POLLING` a `true` mediante `os.environ`. Esto soluciona el error "inotify watch limit reached" que puede aparecer en sistemas con un límite bajo de inotify watchers cuando Streamlit monitoriza archivos.

## Instalación

1. Clona este repositorio.
2. Instala las dependencias de Python:

```bash
pip install -r requirements.txt
