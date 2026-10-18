#!/usr/bin/env python3
"""Tigrinya tokenizer playground — a tiny zero-dependency web app.

A user types Tigrinya text in the browser and sees it broken into BPE
tokens (colored chips with IDs), plus live stats (token count, bytes,
bytes/token, vocab size).

Run:
    python web/server.py
Then open http://localhost:8000

No external web framework is used — only the Python standard library plus
the already-installed `tokenizers` / `tigrinya_tokenizer` packages.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokenizers import Tokenizer  # noqa: E402

from tigrinya_tokenizer import TigrinyaNormalizer  # noqa: E402

TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", REPO_ROOT / "tokenizer.json"))
WEB_DIR = Path(__file__).resolve().parent

if not TOKENIZER_PATH.exists():
    sys.exit(f"Tokenizer not found at {TOKENIZER_PATH}. Train one first.")

_TOK = Tokenizer.from_file(str(TOKENIZER_PATH))
_NORM = TigrinyaNormalizer()
_VOCAB_SIZE = _TOK.get_vocab_size()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path == "/api/info":
            info = {"vocab_size": _VOCAB_SIZE,
                    "tokenizer": str(TOKENIZER_PATH.name)}
            self._send(200, json.dumps(info).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/tokenize":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            text = json.loads(raw).get("text", "")
        except Exception:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        norm = _NORM.normalize(text)
        enc = _TOK.encode(norm, add_special_tokens=False)

        tokens = []
        for tid, tok, (start, end) in zip(enc.ids, enc.tokens, enc.offsets):
            tokens.append({
                "id": int(tid),
                "token": tok,
                "start": int(start),
                "end": int(end),
            })

        n_tokens = len(tokens)
        n_bytes = len(text.encode("utf-8"))
        n_norm_bytes = len(norm.encode("utf-8"))
        out = {
            "text": text,
            "normalized": norm,
            "tokens": tokens,
            "ids": [t["id"] for t in tokens],
            "n_tokens": n_tokens,
            "n_bytes": n_bytes,
            "n_norm_bytes": n_norm_bytes,
            "bytes_per_token": round(n_norm_bytes / n_tokens, 3) if n_tokens else 0.0,
            "vocab_size": _VOCAB_SIZE,
        }
        self._send(200, json.dumps(out).encode("utf-8"), "application/json")

    def log_message(self, *args) -> None:  # silence default logging
        pass


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    srv = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Tigrinya tokenizer playground")
    print(f"  tokenizer : {TOKENIZER_PATH}")
    print(f"  vocab size: {_VOCAB_SIZE:,}")
    print(f"  serving   : {url}")
    print(f"  (Ctrl-C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        srv.shutdown()


if __name__ == "__main__":
    main()
