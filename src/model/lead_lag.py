from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("data/processed/flu_trends_merged.csv")
    summary_output_path = Path("outputs/lead_lag_summary.csv")
    full_output_path = Path("outputs/lead_lag_full.csv")

    keywords = ["flu symptoms", "flu", "fever", "cough", "flu test"]
    target_col = "ili"
    max_lag = 8

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    full_rows: list[dict] = []
    summary_rows: list[dict] = []

    for keyword in keywords:
        lag_results: list[tuple[int, float]] = []
        for lag in range(0, max_lag + 1):
            # lag N means keyword leads ili by N weeks.
            corr = df[target_col].corr(df[keyword].shift(lag))
            lag_results.append((lag, corr))
            full_rows.append(
                {
                    "keyword": keyword,
                    "lag_weeks": lag,
                    "correlation": corr,
                }
            )

        lag_df = pd.DataFrame(lag_results, columns=["lag_weeks", "correlation"])
        valid_lag_df = lag_df.dropna(subset=["correlation"])
        if valid_lag_df.empty:
            best_lag = pd.NA
            best_corr = pd.NA
        else:
            best_idx = valid_lag_df["correlation"].abs().idxmax()
            best_lag = int(valid_lag_df.loc[best_idx, "lag_weeks"])
            best_corr = float(valid_lag_df.loc[best_idx, "correlation"])

        summary_rows.append(
            {
                "keyword": keyword,
                "best_lag_weeks": best_lag,
                "best_correlation": best_corr,
            }
        )

    full_df = pd.DataFrame(full_rows, columns=["keyword", "lag_weeks", "correlation"])
    summary_df = pd.DataFrame(
        summary_rows, columns=["keyword", "best_lag_weeks", "best_correlation"]
    )

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    full_df.to_csv(full_output_path, index=False)

    summary_sorted = summary_df.sort_values(
        by="best_correlation",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)
    print(summary_sorted)


if __name__ == "__main__":
    main()
