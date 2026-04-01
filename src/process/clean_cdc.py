from pathlib import Path

import pandas as pd


def main() -> None:
    raw_path = Path("data/raw/cdc_flu_weekly.csv")
    output_path = Path("data/processed/cdc_clean.csv")

    df = pd.read_csv(raw_path, skiprows=1)

    df = df[["YEAR", "WEEK", "% WEIGHTED ILI"]].rename(
        columns={
            "YEAR": "year",
            "WEEK": "week",
            "% WEIGHTED ILI": "ili",
        }
    )

    for col in ["year", "week", "ili"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values(["year", "week"])

    # Convert to integer year/week after null removal for ISO week parsing.
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)

    iso_week_str = (
        df["year"].astype(str)
        + "-W"
        + df["week"].astype(str).str.zfill(2)
        + "-1"
    )
    df["date"] = pd.to_datetime(iso_week_str, format="%G-W%V-%u")

    df = df[["date", "year", "week", "ili"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())
    print(df.tail())
    print(df.shape)


if __name__ == "__main__":
    main()
