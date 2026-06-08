"""Launch the World Cup Edit Agent web app."""
import webbrowser
import threading

import uvicorn

from app.config import settings


def _open_browser():
    webbrowser.open(f"http://{settings.host}:{settings.port}")


if __name__ == "__main__":
    threading.Timer(1.5, _open_browser).start()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
