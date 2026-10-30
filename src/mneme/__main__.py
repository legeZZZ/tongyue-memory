"""Run the Mneme HTTP service."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8097"))
    uvicorn.run("mneme.app:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    main()
