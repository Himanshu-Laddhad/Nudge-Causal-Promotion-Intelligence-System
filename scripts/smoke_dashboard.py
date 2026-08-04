"""
Execute every dashboard page top-to-bottom and fail on the first exception.

`streamlit run` returning HTTP 200 only proves the shell HTML was served — page
scripts run later over the websocket, so a KeyError in a chart never shows up in
a curl check. This runs each page in Streamlit's bare mode, where st.* calls are
no-ops but the surrounding data code executes for real.

Usage:
    python scripts/smoke_dashboard.py
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAGES = ROOT / 'dashboard' / 'pages'


def main() -> int:
    sys.path.insert(0, str(ROOT))
    warnings.filterwarnings('ignore')

    pages = [ROOT / 'dashboard' / 'app.py'] + sorted(PAGES.glob('*.py'))
    failures: list[tuple[str, Exception]] = []

    for page in pages:
        try:
            runpy.run_path(str(page), run_name='__not_main__')
            print(f'  OK    {page.name}')
        except SystemExit:
            print(f'  OK    {page.name}  (st.stop)')
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            print(f'  FAIL  {page.name}: {type(exc).__name__}: {exc}')
            failures.append((page.name, exc))

    if failures:
        print(f'\n{len(failures)} page(s) failed.')
        return 1
    print(f'\nAll {len(pages)} pages executed cleanly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
