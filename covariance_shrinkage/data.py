"""Download and parse Ken French's data: the 100 size/book-to-market portfolios and the
Fama-French factors (for the risk-free rate), both daily.

Returns are simple returns in decimals, missing values NaN.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
URL = FTP + "100_Portfolios_10x10_Daily_CSV.zip"
FACTORS_URL = FTP + "F-F_Research_Data_Factors_daily_CSV.zip"
VALUE_WEIGHTED = "Average Value Weighted Returns -- Daily"
EQUAL_WEIGHTED = "Average Equal Weighted Returns -- Daily"
MISSING_CODES = (-99.99, -999.0)


def _parse_block(lines: list[str], header: int) -> pd.DataFrame:
    """Parse the block whose header row (starting with a comma) is lines[header].

    Rows ``YYYYMMDD, v1, v2, ...`` follow until a blank line or the end of the file. Values
    are percent; -99.99 and -999 mean missing.
    """
    names = [name.strip() for name in lines[header].split(",")[1:]]
    dates, rows = [], []
    for line in lines[header + 1 :]:
        if not line.strip():
            break
        date, *values = line.split(",")
        dates.append(date.strip())
        rows.append([float(v) for v in values])

    values = np.array(rows)
    if values.shape[1] != len(names):
        raise ValueError(f"{values.shape[1]} values per row but {len(names)} column names")
    values[np.isin(values, MISSING_CODES)] = np.nan
    index = pd.to_datetime(dates, format="%Y%m%d")
    return pd.DataFrame(values / 100.0, index=index, columns=names)


def parse_french_daily(text: str, section: str = VALUE_WEIGHTED) -> pd.DataFrame:
    """Parse one section of a Ken French daily CSV.

    The file holds several sections, each a title line, a header row starting with a
    comma, then rows ``YYYYMMDD, v1, v2, ...`` until a blank line. Values are percent;
    -99.99 and -999 mean missing.

    Returns a DataFrame indexed by date (DatetimeIndex), one column per portfolio,
    simple returns in decimals, NaN where missing.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == section)
    except StopIteration:
        raise ValueError(f"section {section!r} not found") from None
    return _parse_block(lines, start + 1)


def parse_french_factors_daily(text: str) -> pd.DataFrame:
    """Parse the daily Fama-French factors file (F-F_Research_Data_Factors_daily).

    The file has a few lines of description, then a single block: a header row
    ``,Mkt-RF,SMB,HML,RF`` and rows ``YYYYMMDD, v1, ...`` until a blank line. RF is the
    daily simple rate that compounds to the one-month T-bill rate over the month's trading
    days; French publishes it in percent with two decimals, i.e. rounded to 0.01% a day.

    Returns a DataFrame indexed by date, columns Mkt-RF, SMB, HML, RF, in decimals.
    """
    lines = text.splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    except StopIteration:
        raise ValueError("header row ',Mkt-RF,...' not found") from None
    return _parse_block(lines, header)


def _read_cached_zip(url: str, cache_dir: Path | str, refresh: bool) -> str:
    """The text of the single CSV in the zip at url, downloaded once into cache_dir."""
    cache = Path(cache_dir) / url.rsplit("/", 1)[1]
    if refresh or not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            cache.write_bytes(response.read())
    with zipfile.ZipFile(cache) as archive:
        (name,) = archive.namelist()
        return io.TextIOWrapper(archive.open(name), encoding="latin-1").read()


def load_french_100_daily(
    cache_dir: Path | str = "data", section: str = VALUE_WEIGHTED, refresh: bool = False
) -> pd.DataFrame:
    """The 100 size/BM portfolios, daily (value-weighted by default).

    Downloads the zip once into ``cache_dir`` and reads it from there afterwards;
    ``refresh=True`` downloads it again (French updates the file monthly).
    """
    return parse_french_daily(_read_cached_zip(URL, cache_dir, refresh), section)


def load_french_factors_daily(cache_dir: Path | str = "data", refresh: bool = False) -> pd.DataFrame:
    """The daily Fama-French factors, including the risk-free rate RF; cached as above."""
    return parse_french_factors_daily(_read_cached_zip(FACTORS_URL, cache_dir, refresh))
