"""Download Premier League season files from Football-Data.co.uk into ``data/``.

Usage:
    python fetch_data.py                # seasons 2014/15 → 2025/26
    python fetch_data.py 1920 2526      # an explicit range of season codes
    python fetch_data.py --force        # re-download files that already exist
"""

from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://football-data.co.uk/mmz4281/{code}/E0.csv"
DATA_DIR = Path("data")
FIRST_SEASON = 2014
LAST_SEASON = 2025


def season_code(start_year: int) -> str:
    """Start year → Football-Data code: 2014 → ``1415``."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_codes(first: int, last: int) -> list[str]:
    return [season_code(year) for year in range(first, last + 1)]


def download_csv(url: str, attempts: int = 4, wait: float = 15.0) -> bytes:
    """Fetch a Football-Data CSV.

    When busy, the site answers 200 with an HTML "temporarily unavailable" page, so
    the body is checked for the CSV header and the request retried with backoff.
    """
    problem = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "ml-football-predictions"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
            if body.removeprefix(b"\xef\xbb\xbf").startswith(b"Div,"):
                return body
            problem = "response is not a Football-Data CSV"
        except OSError as exc:  # URLError, HTTPError and timeouts
            problem = str(exc)
        if attempt < attempts:
            time.sleep(wait * attempt)
    raise RuntimeError(f"{url}: {problem} (after {attempts} attempts)")


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
        target.write_bytes(download_csv(BASE_URL.format(code=code)))
        print(f"saved  {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
