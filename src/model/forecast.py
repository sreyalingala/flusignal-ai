from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main() -> None:
    input_path = Path("data/processed/flu_trends_merged.csv")
    predictions_output_path = Path("outputs/forecast_predictions.csv")
    metrics_output_path = Path("outputs/forecast_metrics.txt")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Lead-lag baseline features from prior analysis.
    df["flu_symptoms_lag1"] = df["flu symptoms"].shift(1)
    df["flu_lag1"] = df["flu"].shift(1)
    df["flu_test_lag0"] = df["flu test"].shift(0)

    model_df = df[["date", "ili", "flu_symptoms_lag1", "flu_lag1", "flu_test_lag0"]].dropna()

    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    feature_cols = ["flu_symptoms_lag1", "flu_lag1", "flu_test_lag0"]
    target_col = "ili"

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "actual_ili": y_test.values,
            "predicted_ili": y_pred,
        }
    )

    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(predictions_output_path, index=False)

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(f"RMSE: {rmse:.6f}\nR2: {r2:.6f}\n", encoding="utf-8")

    print(f"RMSE: {rmse:.6f}")
    print(f"R2: {r2:.6f}")
    print(pred_df.head())


if __name__ == "__main__":
    main()
