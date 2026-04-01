from pathlib import Path

import pandas as pd


def main() -> None:
    cdc_path = Path("data/processed/cdc_clean.csv")
    trends_path = Path("data/raw/google_trends_weekly.csv")
    output_path = Path("data/processed/flu_trends_merged.csv")

    cdc_df = pd.read_csv(cdc_path)
    trends_df = pd.read_csv(trends_path)

    cdc_df["date"] = pd.to_datetime(cdc_df["date"], errors="coerce")
    trends_df["date"] = pd.to_datetime(trends_df["date"], errors="coerce")

    cdc_iso = cdc_df["date"].dt.isocalendar()
    cdc_df["iso_year"] = cdc_iso.year
    cdc_df["iso_week"] = cdc_iso.week
    cdc_df = cdc_df[["date", "year", "week", "ili", "iso_year", "iso_week"]]

    trends_iso = trends_df["date"].dt.isocalendar()
    trends_df["iso_year"] = trends_iso.year
    trends_df["iso_week"] = trends_iso.week
    trends_df = trends_df[
        [
            "date",
            "flu symptoms",
            "flu",
            "fever",
            "cough",
            "flu test",
            "iso_year",
            "iso_week",
        ]
    ]
    trends_df = trends_df.rename(columns={"date": "trends_date"})

    df = pd.merge(cdc_df, trends_df, on=["iso_year", "iso_week"], how="inner")
    df = df[
        [
            "date",
            "trends_date",
            "year",
            "week",
            "iso_year",
            "iso_week",
            "ili",
            "flu symptoms",
            "flu",
            "fever",
            "cough",
            "flu test",
        ]
    ]
    df = df.sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())
    print(df.tail())
    print(df.shape)
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
