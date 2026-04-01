from pathlib import Path

import pandas as pd


def main() -> None:
    input_path = Path("outputs/forecast_predictions.csv")
    output_path = Path("outputs/alerts.csv")

    df = pd.read_csv(input_path)

    def get_alert(predicted_ili: float) -> str:
        if predicted_ili > 3:
            return "High Risk"
        if 2 <= predicted_ili <= 3:
            return "Moderate Risk"
        return "Low Risk"

    df["alert"] = df["predicted_ili"].apply(get_alert)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.head())


if __name__ == "__main__":
    main()
