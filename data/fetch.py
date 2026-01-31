"""
Data fetching module for Prague housing analyzer.

Data sources:
- Housing prices: ČSÚ (Czech Statistical Office) via Eurostat HPI proxy
- Mortgage rates: CNB ARAD database
- FX rates: CNB exchange rate API
- Equity returns: Yahoo Finance (S&P 500, PX index)
"""

import json
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
import polars as pl

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_EXPIRY_HOURS = 24


def _get_cache_path(key: str) -> Path:
    """Get cache file path for a given key."""
    CACHE_DIR.mkdir(exist_ok=True)
    hashed = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{key}_{hashed}.parquet"


def _is_cache_valid(cache_path: Path, expiry_hours: int = CACHE_EXPIRY_HOURS) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=expiry_hours)


def _read_cache(cache_path: Path) -> Optional[pl.DataFrame]:
    """Read DataFrame from cache if valid."""
    if _is_cache_valid(cache_path):
        try:
            return pl.read_parquet(cache_path)
        except Exception:
            return None
    return None


def _write_cache(cache_path: Path, df: pl.DataFrame) -> None:
    """Write DataFrame to cache."""
    try:
        df.write_parquet(cache_path)
    except Exception:
        pass  # Silently ignore cache write failures


def fetch_housing_prices(use_cache: bool = True) -> pl.DataFrame:
    """
    Fetch Prague historical housing prices.
    
    Uses Eurostat House Price Index (HPI) data for Czech Republic as proxy,
    since granular Prague-level data requires complex ČSÚ API navigation.
    
    Returns:
        DataFrame with columns: date, hpi_index (2015=100), yoy_change_pct
    """
    cache_path = _get_cache_path("housing_prices")
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    
    # Eurostat HPI API v1.0 - House Price Index for Czech Republic
    # Dataset: prc_hpi_q (quarterly house price indices)
    # Unit I15_Q = Index 2015=100
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        "prc_hpi_q?freq=Q&unit=I15_Q&purchase=TOTAL&geo=CZ&format=JSON"
    )
    
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        
        # Parse Eurostat JSON-stat 2.0 format
        time_index = data["dimension"]["time"]["category"]["index"]
        values = data["value"]
        
        # In this format, values dict maps flat index to value
        # We need to find which index corresponds to each time period
        records = []
        for period, idx in time_index.items():
            val_key = str(idx)
            if val_key in values:
                # Parse quarter format like "2015-Q1"
                year, quarter = period.split("-Q")
                month = (int(quarter) - 1) * 3 + 1
                records.append({
                    "date": date(int(year), month, 1),
                    "hpi_index": float(values[val_key]),
                })
        
        df = pl.DataFrame(records).sort("date")
        
        # Calculate year-over-year change (4 quarters = 1 year)
        df = df.with_columns(
            ((pl.col("hpi_index") / pl.col("hpi_index").shift(4) - 1) * 100).alias("yoy_change_pct")
        )
        
        _write_cache(cache_path, df)
        return df
        
    except Exception as e:
        # Return empty DataFrame with correct schema on failure
        return pl.DataFrame({
            "date": pl.Series([], dtype=pl.Date),
            "hpi_index": pl.Series([], dtype=pl.Float64),
            "yoy_change_pct": pl.Series([], dtype=pl.Float64),
        })


def fetch_cnb_rates(use_cache: bool = True) -> pl.DataFrame:
    """
    Fetch Czech mortgage interest rates.
    
    Uses OECD long-term interest rates for Czech Republic as base,
    with adjustment to approximate mortgage rates.
    
    Returns:
        DataFrame with columns: date, mortgage_rate_pct
    """
    cache_path = _get_cache_path("cnb_rates")
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    
    # OECD MEI_FIN - Long-term interest rates for Czech Republic
    # IRLT = Long-term interest rates (government bonds)
    # Add ~1.5% spread to approximate mortgage rates
    url = (
        "https://stats.oecd.org/sdmx-json/data/MEI_FIN/IRLT.CZE.M/all"
        "?startTime=2010&endTime=2026"
    )
    
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
        
        # Parse SDMX-JSON 2.0 format
        # Structure is under data.structures[0].dimensions.observation
        data_obj = data.get("data", {})
        structures = data_obj.get("structures", [])
        if not structures:
            raise ValueError("No structures in OECD response")
        
        obs_dims = structures[0].get("dimensions", {}).get("observation", [])
        time_periods = obs_dims[0].get("values", []) if obs_dims else []
        
        # Series data is under data.dataSets[0].series
        datasets = data_obj.get("dataSets", [])
        if not datasets:
            raise ValueError("No datasets in OECD response")
        
        all_series = datasets[0].get("series", {})
        
        records = []
        for series_key, series_data in all_series.items():
            obs = series_data.get("observations", {})
            for time_idx_str, values in obs.items():
                if values and len(values) > 0 and values[0] is not None:
                    time_idx = int(time_idx_str)
                    if time_idx < len(time_periods):
                        time_info = time_periods[time_idx]
                        time_str = time_info.get("id", "")  # Format: "2010-01"
                        try:
                            d = datetime.strptime(time_str + "-01", "%Y-%m-%d").date()
                            # Filter to 2010+
                            if d.year >= 2010:
                                # Add spread to get approximate mortgage rate
                                rate = float(values[0]) + 1.5
                                records.append({
                                    "date": d,
                                    "mortgage_rate_pct": rate,
                                })
                        except (ValueError, TypeError):
                            continue
        
        if records:
            # Deduplicate and sort
            df = pl.DataFrame(records).unique(subset=["date"]).sort("date")
            _write_cache(cache_path, df)
            return df
            
    except Exception:
        pass
    
    # Return empty DataFrame with correct schema
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "mortgage_rate_pct": pl.Series([], dtype=pl.Float64),
    })


def fetch_fx_rates(use_cache: bool = True) -> pl.DataFrame:
    """
    Fetch USD/CZK exchange rate history from CNB.
    
    Returns:
        DataFrame with columns: date, usd_czk
    """
    cache_path = _get_cache_path("fx_rates")
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    
    # CNB exchange rates - historical daily data
    # Format: year by year to get 10+ years of data
    current_year = datetime.now().year
    all_records = []
    
    try:
        with httpx.Client(timeout=30) as client:
            for year in range(current_year - 12, current_year + 1):
                url = (
                    f"https://www.cnb.cz/en/financial-markets/foreign-exchange-market/"
                    f"central-bank-exchange-rate-fixing/central-bank-exchange-rate-fixing/"
                    f"year.txt?year={year}"
                )
                
                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        continue
                    text = resp.text
                    
                    lines = text.strip().split("\n")
                    if len(lines) < 2:
                        continue
                    
                    # First line is header with currency codes (Date|1 AUD|...|1 USD|...)
                    header = lines[0].split("|")
                    usd_idx = None
                    for i, col in enumerate(header):
                        if "USD" in col:
                            usd_idx = i
                            break
                    
                    if usd_idx is None:
                        continue
                    
                    # Data lines start from line 1
                    for line in lines[1:]:
                        parts = line.split("|")
                        if len(parts) > usd_idx:
                            try:
                                d = datetime.strptime(parts[0].strip(), "%d.%m.%Y").date()
                                rate = float(parts[usd_idx].strip().replace(",", "."))
                                all_records.append({"date": d, "usd_czk": rate})
                            except (ValueError, IndexError):
                                continue
                                
                except httpx.HTTPError:
                    continue
        
        if all_records:
            df = pl.DataFrame(all_records).sort("date").unique(subset=["date"])
            _write_cache(cache_path, df)
            return df
            
    except Exception:
        pass
    
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "usd_czk": pl.Series([], dtype=pl.Float64),
    })


def fetch_equity_returns(use_cache: bool = True) -> pl.DataFrame:
    """
    Fetch historical equity index data (S&P 500 and PX index).
    
    Uses Yahoo Finance for S&P 500 and Stooq for PX index.
    
    Returns:
        DataFrame with columns: date, sp500_close, px_close
    """
    cache_path = _get_cache_path("equity_returns")
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
    
    def fetch_yahoo_data(symbol: str) -> dict[date, float]:
        """Fetch historical data from Yahoo Finance."""
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=365*12)).timestamp())
        
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d"
        )
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        
        result = data.get("chart", {}).get("result", [])
        if not result:
            return {}
        
        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        
        records = {}
        for ts, close in zip(timestamps, closes):
            if ts and close:
                d = datetime.fromtimestamp(ts).date()
                records[d] = float(close)
        
        return records
    
    def fetch_stooq_data(symbol: str) -> dict[date, float]:
        """Fetch historical data from Stooq (better for PX index)."""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=365*12)).strftime("%Y%m%d")
        
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d&d1={start_date}&d2={end_date}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            text = resp.text
        
        records = {}
        lines = text.strip().split("\n")
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(",")
            if len(parts) >= 5:
                try:
                    d = datetime.strptime(parts[0], "%Y-%m-%d").date()
                    close = float(parts[4])  # Close is 5th column
                    records[d] = close
                except (ValueError, IndexError):
                    continue
        
        return records
    
    try:
        # Fetch S&P 500 from Yahoo Finance
        sp500_data = fetch_yahoo_data("^GSPC")
        
        # Fetch PX index from Stooq (better historical coverage)
        px_data = fetch_stooq_data("^px")
        
        # Merge on dates (use all dates from both)
        all_dates = sorted(set(sp500_data.keys()) | set(px_data.keys()))
        
        records = []
        for d in all_dates:
            records.append({
                "date": d,
                "sp500_close": sp500_data.get(d),
                "px_close": px_data.get(d),
            })
        
        if records:
            df = pl.DataFrame(records).sort("date")
            _write_cache(cache_path, df)
            return df
            
    except Exception:
        pass
    
    return pl.DataFrame({
        "date": pl.Series([], dtype=pl.Date),
        "sp500_close": pl.Series([], dtype=pl.Float64),
        "px_close": pl.Series([], dtype=pl.Float64),
    })


def fetch_all_data(use_cache: bool = True) -> dict[str, pl.DataFrame]:
    """
    Fetch all data sources and return as dictionary.
    
    Args:
        use_cache: Whether to use cached data if available.
        
    Returns:
        Dictionary with keys:
        - 'housing_prices': Prague/CZ housing price index
        - 'mortgage_rates': CNB mortgage interest rates
        - 'fx_rates': USD/CZK exchange rates
        - 'equity_returns': S&P 500 and PX index data
    """
    return {
        "housing_prices": fetch_housing_prices(use_cache=use_cache),
        "mortgage_rates": fetch_cnb_rates(use_cache=use_cache),
        "fx_rates": fetch_fx_rates(use_cache=use_cache),
        "equity_returns": fetch_equity_returns(use_cache=use_cache),
    }


if __name__ == "__main__":
    # Quick test
    print("Fetching all data...")
    data = fetch_all_data(use_cache=False)
    for name, df in data.items():
        print(f"\n{name}: {len(df)} rows")
        if len(df) > 0:
            print(df.head(3))
