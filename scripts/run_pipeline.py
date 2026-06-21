"""Convenience entry point: `python scripts/run_pipeline.py ...`

Just delegates to coastal_pinn.cli.main() with sys.argv. Exists so that
users who prefer running from the project root don't need to remember
`python -m coastal_pinn`.
"""

import sys

from coastal_pinn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))