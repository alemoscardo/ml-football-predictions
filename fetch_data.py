"""Download Premier League season files from Football-Data.co.uk into ``data/``.

Usage:
    python fetch_data.py                # seasons 2014/15 → 2025/26
    python fetch_data.py 1920 2526      # an explicit range of season codes
    python fetch_data.py --force        # re-download files that already exist
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
DATA_DIR = Path("data")
FIRST_SEASON = 2014
LAST_SEASON = 2025


def season_codes(first: int, last: int) -> list[str]:
    """Start years → Football-Data codes: 2014 → ``1415``."""
    return [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(first, last + 1)]


def main(argv: list[str]) -> None:
    force = "--force" in argv
    args = [a for a in argv if not a.startswith("--")]
    first, last = FIRST_SEASON, LAST_SEASON
    if len(args) == 2:
        first, last = (2000 + int(code[:2]) for code in args)

    DATA_DIR.mkdir(exist_ok=True)
    for code in season_codes(first, last):
        target = DATA_DIR / f"E0_{code}.csv"
        if target.exists() and not force:
            print(f"skip   {target} (exists)")
            continue
        request = urllib.request.Request(
            BASE_URL.format(code=code), headers={"User-Agent": "ml-football-predictions"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            target.write_bytes(response.read())
        print(f"saved  {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
