# Lista Bookeo Interactiva

This Streamlit app lets you manage visit lists generated from Bookeo PDFs.
The very first lines of `app.py` set the environment variable
`WATCHDOG_USE_POLLING` to `true` via `os.environ`. This mitigates the
"inotify watch limit reached" error that may appear when Streamlit watches
files on systems with a low limit of inotify watchers.

Run the app with:

```bash
streamlit run app.py
```

