"""Compatibility launcher for the OpenMedVisionX GUI.

Prefer the installed ``openmedvisionx`` command.  Keeping this tiny wrapper
allows existing course notes that use ``python main.py`` to continue working
without reintroducing UI, IO, or algorithm code into the entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dicom_viewer.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
