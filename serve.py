import http.server
import socketserver
import webbrowser
import threading
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}/index.html')

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()
