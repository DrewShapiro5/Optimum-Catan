#!/usr/bin/env python3
"""Open the Catanatron starting menu in the browser."""

import argparse
from pathlib import Path
import threading
import webbrowser

from flask import abort, send_from_directory
from werkzeug.serving import make_server

from catanatron.web import create_app


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "framework" / "ui" / "dist"


def make_app(database_uri=None):
    if not (DIST / "index.html").is_file():
        raise RuntimeError("Browser interface is not built. Run ./setup.sh first.")
    if database_uri is None:
        data = ROOT / "data"
        data.mkdir(exist_ok=True)
        database_uri = f"sqlite:///{data / 'games.sqlite3'}"
    app = create_app({"SQLALCHEMY_DATABASE_URI": database_uri})
    from agent_apis.web import install
    install(app)

    @app.get("/", defaults={"path": ""})
    @app.get("/<path:path>")
    def frontend(path):
        if path == "api" or path.startswith("api/"):
            abort(404)
        if path and (DIST / path).is_file():
            return send_from_directory(DIST, path)
        if not path or path.startswith(("games/", "replays/")):
            return send_from_directory(DIST, "index.html")
        abort(404)

    return app


def new_game(app):
    # The upstream RL examples require external trained model files. None are
    # shipped, so use the requested two-human fallback (shared-screen play).
    with app.test_client() as client:
        response = client.post("/api/games", json={
            "players": ["HUMAN", "HUMAN"],
            "map_template": "BASE",
            "vps_to_win": 15,
            "discard_limit": 9,
            "friendly_robber": True,
        })
        if response.status_code != 200:
            raise RuntimeError(f"Could not create game: {response.get_data(as_text=True)}")
        return response.get_json()["game_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quick-start", action="store_true", help="Skip the menu and create a two-human game")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    app = make_app()
    # Bind first, so an occupied port cannot open an unrelated server's page.
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    path = f"/games/{new_game(app)}" if args.quick_start else "/"
    url = f"http://127.0.0.1:{server.server_port}{path}"
    print("Choose 2 or 4 players and agent APIs in the starting menu.", flush=True)
    print("Humans share the browser with open hands; controls follow the active seat.", flush=True)
    print(f"Catanatron: {url}", flush=True)
    print("Press Ctrl+C to stop. Games are saved in data/games.sqlite3.", flush=True)
    if not args.no_browser:
        def open_browser():
            if not webbrowser.open(url):
                print(f"Open this link in your browser: {url}", flush=True)
        # The socket is already listening; the request waits until serve_forever.
        threading.Thread(target=open_browser, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
