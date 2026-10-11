"""Console entry point that launches the packaged Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch DataChef through Streamlit while forwarding CLI arguments."""
    from streamlit.web import cli as streamlit_cli

    # Streamlit's help includes Unicode glyphs. Older Windows console encodings
    # can otherwise raise UnicodeEncodeError before the server even starts.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
