from pathlib import Path
import time

import pandas as pd
from pytrends.request import TrendReq


def fetch_window(
    pytrends: TrendReq, keywords: list[str], geo: str, window_timeframe: str
) -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            pytrends.build_payload(
                keywords, cat=0, timeframe=window_timeframe, geo=geo, gprop=""
            )
            return pytrends.interest_over_time()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    return pd.DataFrame()


def main() -> None:
    keywords = ["flu symptoms", "flu", "fever", "cough", "flu test"]
    geo = "US"
    start = pd.Timestamp("2016-10-01")
    end = pd.Timestamp("2026-03-15")

    pytrends = TrendReq(hl="en-US", tz=360)

    dfs: list[pd.DataFrame] = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + pd.DateOffset(years=1), end)
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"

        window_df = fetch_window(pytrends, keywords, geo, timeframe)
        if not window_df.empty:
            dfs.append(window_df)

        current_start = current_end
        time.sleep(5)

    if dfs:
        df = pd.concat(dfs)
    else:
        df = pd.DataFrame(columns=["date"] + keywords)

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    if not df.empty:
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        df = df.reset_index()
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date")
        df = df[keywords].resample("W-SUN").last()
        df = df.reset_index()
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
    else:
        df = pd.DataFrame(columns=["date"] + keywords)

    output_path = Path("data/raw/google_trends_weekly.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())
    print(df.tail())
    print(df.shape)


if __name__ == "__main__":
    main()
