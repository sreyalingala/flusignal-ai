from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    input_path = Path("outputs/forecast_predictions.csv")
    output_path = Path("outputs/forecast_plot.png")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"])

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["actual_ili"], label="actual_ili")
    plt.plot(df["date"], df["predicted_ili"], label="predicted_ili")
    plt.title("Flu Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("ILI")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
