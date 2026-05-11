# 01 — Data Collection & Sources

> **Module**: 02 Data Engineering | **File**: 1 of 6
>
> Before any analysis or modeling, you need to reliably load, parse, and structure time series data. This file covers every common data source and the correct patterns for handling them.

---

## Table of Contents

1. [Data Sources Overview](#1-data-sources-overview)
2. [Loading CSV and Parquet Files](#2-loading-csv-and-parquet-files)
3. [Datetime Parsing and DatetimeIndex](#3-datetime-parsing-and-datetimeindex)
4. [Loading from APIs](#4-loading-from-apis)
5. [Loading from Databases](#5-loading-from-databases)
6. [Validating a Time Series After Loading](#6-validating-a-time-series-after-loading)
7. [Common Pitfalls](#7-common-pitfalls)

---

## 1. Data Sources Overview

| Source | Format | Best Library | Use Case |
|--------|--------|-------------|----------|
| Flat files | CSV, TSV | `pandas` | Most common — local or cloud storage |
| Columnar storage | Parquet, Feather | `pandas`, `pyarrow` | Large datasets, fast I/O |
| Financial markets | REST API | `yfinance`, `pandas-datareader` | Stock/FX/crypto prices |
| Public datasets | REST API | `requests`, `pandas` | Weather, energy, economic data |
| SQL databases | SQL | `sqlalchemy`, `pandas` | Enterprise data warehouses |
| Time-series databases | InfluxDB, TimescaleDB | InfluxDB client, `psycopg2` | IoT, sensor, operational data |
| Cloud storage | S3, GCS, Azure Blob | `pandas`, `s3fs`, `gcsfs` | Production pipelines |
| Streaming | Kafka, Kinesis | `kafka-python`, `boto3` | Real-time data ingestion |

---

## 2. Loading CSV and Parquet Files

### 2.1 CSV — Basic Loading

```python
import pandas as pd

# ── Option 1: parse_dates at load time (preferred) ──────────────────────────
df = pd.read_csv(
    "data/sales.csv",
    parse_dates=["date"],      # columns to parse as datetime
    index_col="date",          # set as index immediately
    dayfirst=False,            # True if dates are DD/MM/YYYY format
)

# ── Option 2: parse after loading ───────────────────────────────────────────
df = pd.read_csv("data/sales.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

print(df.head())
print(f"\nIndex type : {type(df.index)}")
print(f"Date range : {df.index.min()} → {df.index.max()}")
print(f"Shape      : {df.shape}")
```

### 2.2 CSV — Common Format Issues

```python
# European date format: DD.MM.YYYY
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

# ISO 8601 with timezone
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

# Mixed formats (slow but handles inconsistency)
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)

# Unix timestamps (seconds since epoch)
df["date"] = pd.to_datetime(df["timestamp_sec"], unit="s")

# Millisecond timestamps
df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
```

### 2.3 Parquet — Recommended for Large Datasets

```python
# Writing Parquet (preserves dtypes including datetime)
df.to_parquet("data/sales.parquet", engine="pyarrow", compression="snappy")

# Reading Parquet (much faster than CSV for large files)
df = pd.read_parquet("data/sales.parquet", engine="pyarrow")

# Read only specific columns (columnar format advantage)
df = pd.read_parquet("data/sales.parquet", columns=["value", "category"])
```

**Parquet vs CSV comparison:**

| Property | CSV | Parquet |
|----------|-----|---------|
| Human readable | ✅ Yes | ❌ No (binary) |
| File size | Large | 3–10× smaller |
| Read speed | Slow | Very fast |
| Type preservation | ❌ Loses datetime types | ✅ Preserves all types |
| Column pruning | ❌ Always reads all | ✅ Read only needed columns |
| Recommended for | Config, small data | Production pipelines |

---

## 3. Datetime Parsing and DatetimeIndex

### 3.1 Why DatetimeIndex Matters

A proper `DatetimeIndex` unlocks all time series functionality in pandas:
- Slicing by date range: `df["2023-01":"2023-06"]`
- Resampling: `df.resample("W").sum()`
- Frequency inference and offset arithmetic
- `shift()`, `diff()`, rolling operations with date-aware windows

```python
# Set DatetimeIndex if not already set
df.index = pd.DatetimeIndex(df.index)
df.index.name = "date"

# Verify
print(type(df.index))              # pandas.DatetimeIndex
print(df.index.dtype)             # datetime64[ns]
print(df.index.freq)              # None or 'MS', 'D', 'h', etc.
```

### 3.2 Inferring and Setting Frequency

```python
# Infer frequency automatically (works when series has no gaps)
df.index.freq = pd.infer_freq(df.index)
print(f"Inferred frequency: {df.index.freq}")

# Set frequency manually
df = df.asfreq("D")      # Daily
df = df.asfreq("MS")     # Monthly Start
df = df.asfreq("h")      # Hourly
df = df.asfreq("15min")  # 15-minute
```

### 3.3 Common Frequency Aliases

| Alias | Frequency | Description |
|-------|-----------|-------------|
| `"T"` or `"min"` | Minute | Minute-level |
| `"h"` or `"H"` | Hourly | Hour-level |
| `"D"` | Daily | Calendar day |
| `"B"` | Business Day | Mon–Fri only |
| `"W"` | Weekly | Sunday end of week |
| `"MS"` | Month Start | First day of each month |
| `"ME"` or `"M"` | Month End | Last day of each month |
| `"QS"` | Quarter Start | First day of quarter |
| `"AS"` or `"YS"` | Year Start | January 1st |

### 3.4 Date Range Generation

```python
# Generate a complete date range
idx = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
print(f"Generated {len(idx)} daily timestamps")

# Generate N periods
idx = pd.date_range(start="2023-01-01", periods=365, freq="D")

# Business days only
idx_biz = pd.bdate_range(start="2023-01-01", end="2023-12-31")

# Reindex to fill missing timestamps
df_complete = df.reindex(idx)   # introduces NaN where data is missing
```

---

## 4. Loading from APIs

### 4.1 Yahoo Finance (Financial Data)

```python
import yfinance as yf

# Download daily stock data
df = yf.download(
    tickers="AAPL",
    start="2020-01-01",
    end="2024-01-01",
    interval="1d",        # 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
    auto_adjust=True,     # adjust for splits and dividends
)
print(df.head())
# Columns: Open, High, Low, Close, Volume

# Multiple tickers
df_multi = yf.download(["AAPL", "GOOGL", "MSFT"], start="2022-01-01")
close = df_multi["Close"]
```

### 4.2 Generic REST API with requests

```python
import requests
import pandas as pd

def fetch_timeseries_api(url: str, params: dict, date_col: str, value_col: str) -> pd.Series:
    """Generic REST API fetcher → pandas Series."""
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df[value_col].astype(float)

# Example: Open-Meteo weather API (free, no key needed)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": "temperature_2m_max",
    "timezone": "Asia/Kolkata",
}
response = requests.get(url, params=params)
data = response.json()
temp_series = pd.Series(
    data["daily"]["temperature_2m_max"],
    index=pd.to_datetime(data["daily"]["time"]),
    name="max_temp_delhi"
)
print(temp_series.head())
```

### 4.3 pandas-datareader (Economic / Macro Data)

```python
import pandas_datareader as pdr

# World Bank data
gdp = pdr.get_data_wb("NY.GDP.MKTP.CD", start=2000, end=2023, country="IN")

# FRED (Federal Reserve Economic Data)
inflation = pdr.get_data_fred("CPIAUCSL", start="2010-01-01", end="2023-12-31")
```

---

## 5. Loading from Databases

### 5.1 SQL Database via pandas

```python
import pandas as pd
from sqlalchemy import create_engine

# Create engine (replace with your credentials)
engine = create_engine("postgresql+psycopg2://user:password@host:5432/dbname")

# Load with time filter pushed to SQL (efficient)
query = """
    SELECT timestamp, sensor_id, value
    FROM sensor_readings
    WHERE timestamp BETWEEN '2023-01-01' AND '2023-12-31'
      AND sensor_id = 'TEMP_001'
    ORDER BY timestamp ASC
"""
df = pd.read_sql(query, con=engine, parse_dates=["timestamp"], index_col="timestamp")
print(df.head())
```

### 5.2 InfluxDB (Time-Series Database)

```python
from influxdb_client import InfluxDBClient

client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
query_api = client.query_api()

query = '''
  from(bucket: "sensors")
    |> range(start: -30d)
    |> filter(fn: (r) => r["_measurement"] == "temperature")
    |> aggregateWindow(every: 1h, fn: mean)
'''
df = query_api.query_data_frame(query)
df = df.set_index("_time").sort_index()
```

---

## 6. Validating a Time Series After Loading

Always run these checks immediately after loading:

```python
def validate_timeseries(series: pd.Series, expected_freq: str = None) -> None:
    """Standard validation checks for a loaded time series."""
    print("=" * 50)
    print("  Time Series Validation Report")
    print("=" * 50)

    # Basic info
    print(f"\n  Name       : {series.name}")
    print(f"  Length     : {len(series)} observations")
    print(f"  Start      : {series.index.min()}")
    print(f"  End        : {series.index.max()}")
    print(f"  Index type : {type(series.index).__name__}")
    print(f"  Dtype      : {series.dtype}")

    # Frequency check
    inferred = pd.infer_freq(series.index)
    print(f"\n  Inferred freq : {inferred}")
    if expected_freq and inferred != expected_freq:
        print(f"  ⚠️  Expected {expected_freq}, got {inferred} — check for gaps!")

    # Missing values
    n_missing = series.isna().sum()
    pct_missing = n_missing / len(series) * 100
    print(f"\n  Missing values : {n_missing} ({pct_missing:.2f}%)")

    # Duplicates
    n_dup_idx = series.index.duplicated().sum()
    print(f"  Duplicate timestamps : {n_dup_idx}")

    # Monotonic index
    print(f"  Index monotonic increasing : {series.index.is_monotonic_increasing}")

    # Value statistics
    print(f"\n  Value Statistics:")
    print(series.describe().round(3).to_string())
    print("=" * 50)

# Usage
validate_timeseries(df["Close"], expected_freq="B")
```

**Key things to check:**

| Check | Why It Matters |
|-------|----------------|
| Index is DatetimeIndex | Required for resampling, shifting, rolling |
| Index is monotonically increasing | Models assume time order |
| No duplicate timestamps | Causes silent bugs in resampling |
| Inferred frequency matches expected | Gaps = NaN after `.asfreq()` |
| Missing value count | Determines imputation strategy needed |
| Value range sanity | Negative sales, 0 temperature — domain checks |

---

## 7. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Wrong datetime format** | `NaT` values everywhere after parsing | Specify `format="%Y-%m-%d"` explicitly |
| **Mixed timezones** | Wrong comparisons across series | Normalize to UTC early; convert to local for display |
| **String index instead of DatetimeIndex** | Resampling fails | `df.index = pd.to_datetime(df.index)` |
| **Unsorted index** | Rolling windows and lag features give wrong results | `df = df.sort_index()` |
| **Duplicate timestamps** | Resampling gives wrong counts | `df = df[~df.index.duplicated(keep='first')]` |
| **Loading too much data** | Memory error | Use `usecols`, date filters in SQL, chunked reading |
| **Not setting freq** | `shift()` and `date_range` produce errors | `df = df.asfreq('D')` after loading |

---

*← [Module README](./README.md) | Next: [02 — Resampling & Frequency](./02_resampling_and_frequency.md) →*
