# client/tui/__main__.py
from .app import HephiaTUIApp

if __name__ == "__main__":
    app = HephiaTUIApp()
    app.run()