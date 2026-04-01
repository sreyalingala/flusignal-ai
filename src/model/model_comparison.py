from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
    input_path = Path("data/processed/flu_trends_merged.csv")
    comparison_output_path = Path("outputs/model_comparison.csv")
    predictions_output_path = Path("outputs/model_predictions.csv")

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

    df["flu_symptoms_lag1"] = df["flu symptoms"].shift(1)
    df["flu_lag1"] = df["flu"].shift(1)
    df["flu_test_lag0"] = df["flu test"].shift(0)
    df["cough_lag1"] = df["cough"].shift(1)
    df["ili_lag1"] = df["ili"].shift(1)
    df["ili_lag2"] = df["ili"].shift(2)
    df["flu_symptoms_roll3"] = df["flu symptoms"].rolling(window=3).mean()
    df["flu_roll3"] = df["flu"].rolling(window=3).mean()
    df["flu_test_roll3"] = df["flu test"].rolling(window=3).mean()
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    model_df = df[
        [
            "date",
            "ili",
            "flu_symptoms_lag1",
            "flu_lag1",
            "flu_test_lag0",
            "cough_lag1",
            "ili_lag1",
            "ili_lag2",
            "flu_symptoms_roll3",
            "flu_roll3",
            "flu_test_roll3",
            "weekofyear",
            "month",
        ]
    ]

    feature_cols = [
        "flu_symptoms_lag1",
        "flu_lag1",
        "flu_test_lag0",
        "cough_lag1",
        "ili_lag1",
        "ili_lag2",
        "flu_symptoms_roll3",
        "flu_roll3",
        "flu_test_roll3",
        "weekofyear",
        "month",
    ]
    target_col = "ili"

    x_all = model_df[feature_cols].copy()
    y_all = model_df[target_col].copy()
    date_all = model_df["date"].copy()

    x_all = x_all.astype("float64")
    y_all = pd.to_numeric(y_all, errors="coerce").astype("float64")

    print(x_all.describe())
    print(x_all.dtypes)
    print(y_all.dtype)

    x_all = x_all.replace([np.inf, -np.inf], np.nan)
    y_all = y_all.replace([np.inf, -np.inf], np.nan)

    clean_mask = x_all.notnull().all(axis=1) & y_all.notnull()
    x_all = x_all.loc[clean_mask].reset_index(drop=True)
    y_all = y_all.loc[clean_mask].reset_index(drop=True)
    date_all = date_all.loc[clean_mask].reset_index(drop=True)

    print(f"X has nulls after cleaning: {x_all.isnull().any().any()}")
    print(f"y has nulls after cleaning: {y_all.isnull().any()}")
    print(f"Final dataset shape after full cleaning: {(len(x_all), x_all.shape[1] + 2)}")

    split_idx = int(len(x_all) * 0.8)
    x_train = x_all.iloc[:split_idx]
    y_train = y_all.iloc[:split_idx]
    x_test = x_all.iloc[split_idx:]
    y_test = y_all.iloc[split_idx:]
    test_dates = date_all.iloc[split_idx:]

    zero_var_cols = x_train.columns[x_train.nunique(dropna=True) <= 1].tolist()
    if zero_var_cols:
        x_train = x_train.drop(columns=zero_var_cols)
        x_test = x_test.drop(columns=zero_var_cols)

    final_feature_cols = x_train.columns.tolist()
    if not final_feature_cols:
        raise ValueError("No usable feature columns remain after stability checks.")

    comparison_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    y_train_np = np.asarray(y_train, dtype=np.float64)
    y_test_np = np.asarray(y_test, dtype=np.float64)

    # Manual Ridge
    alpha = 1.0
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = np.asarray(x_train_scaled, dtype=np.float64)
    x_test_scaled = np.asarray(x_test_scaled, dtype=np.float64)

    x_train_finite = np.isfinite(x_train_scaled).all()
    x_test_finite = np.isfinite(x_test_scaled).all()
    y_train_finite = np.isfinite(y_train_np).all()
    y_test_finite = np.isfinite(y_test_np).all()
    print(f"x_train_scaled finite: {x_train_finite}")
    print(f"x_test_scaled finite: {x_test_finite}")
    print(f"y_train_np finite: {y_train_finite}")
    print(f"y_test_np finite: {y_test_finite}")
    if not (x_train_finite and x_test_finite and y_train_finite and y_test_finite):
        raise ValueError("Non-finite values found before ManualRidge calculation.")

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ones_train = np.ones((x_train_scaled.shape[0], 1), dtype=np.float64)
        ones_test = np.ones((x_test_scaled.shape[0], 1), dtype=np.float64)
        x_train_aug = np.hstack([ones_train, x_train_scaled])
        x_test_aug = np.hstack([ones_test, x_test_scaled])

        identity = np.eye(x_train_aug.shape[1], dtype=np.float64)
        identity[0, 0] = 0.0  # Do not regularize intercept.
        beta = np.linalg.solve(
            x_train_aug.T @ x_train_aug + alpha * identity,
            x_train_aug.T @ y_train_np,
        )
        y_pred_manual = x_test_aug @ beta

    comparison_rows.append(
        {
            "model": "ManualRidge",
            "rmse": mean_squared_error(y_test_np, y_pred_manual) ** 0.5,
            "r2": r2_score(y_test_np, y_pred_manual),
        }
    )
    prediction_rows.append(
        pd.DataFrame(
            {
                "date": test_dates.values,
                "model": "ManualRidge",
                "actual_ili": y_test_np,
                "predicted_ili": y_pred_manual,
            }
        )
    )

    tree_models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }
    for model_name, model in tree_models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        comparison_rows.append(
            {
                "model": model_name,
                "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
                "r2": r2_score(y_test, y_pred),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "date": test_dates.values,
                    "model": model_name,
                    "actual_ili": y_test.values,
                    "predicted_ili": y_pred,
                }
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values("r2", ascending=False).reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)

    print(f"Final feature columns used: {final_feature_cols}")
    print(comparison_df)


if __name__ == "__main__":
    main()
