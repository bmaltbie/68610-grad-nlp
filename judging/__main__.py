"""``python -m judging`` entry point."""

from __future__ import annotations

import sys

from judging.cli import main


if __name__ == "__main__":
    sys.exit(main())
