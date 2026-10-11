"""End-to-end retail sales analysis and baseline prediction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "retail_sales.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "sales_inr", "units_sold"])
    df = df.drop_duplicates()
    df["month"] = df["order_date"].dt.month
    df["year"] = df["order_date"].dt.year

    total_sales = float(df["sales_inr"].sum())
    avg_order = float(df["sales_inr"].mean())
    top_category = df.groupby("category")["sales_inr"].sum().idxmax()
    top_region = df.groupby("region")["sales_inr"].sum().idxmax()

    print("=== RETAIL SALES ANALYSIS ===")
    print(f"Rows analysed: {len(df):,}")
    print(f"Total sales: ₹{total_sales:,.2f}")
    print(f"Average order value: ₹{avg_order:,.2f}")
    print(f"Highest-sales category: {top_category}")
    print(f"Highest-sales region: {top_region}")

    monthly = df.set_index("order_date").resample("MS")["sales_inr"].sum()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Monthly Retail Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales (INR)")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / "monthly_sales.png", dpi=150)
    plt.close()

    category_sales = df.groupby("category")["sales_inr"].sum().sort_values()
    plt.figure(figsize=(8, 5))
    category_sales.plot(kind="barh")
    plt.title("Total Sales by Product Category")
    plt.xlabel("Sales (INR)")
    plt.tight_layout()
    plt.savefig(OUT / "category_sales.png", dpi=150)
    plt.close()

    region_sales = df.groupby("region")["sales_inr"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    region_sales.plot(kind="bar")
    plt.title("Sales by Region")
    plt.ylabel("Sales (INR)")
    plt.tight_layout()
    plt.savefig(OUT / "region_sales.png", dpi=150)
    plt.close()

    # Predict sales using order attributes available at order time.
    features = ["region", "category", "units_sold", "unit_price_inr", "discount_pct", "month", "year"]
    X, y = df[features], df["sales_inr"]
    categorical = ["region", "category"]
    numeric = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\n=== BASELINE MODEL (held-out test set) ===")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²: {r2:.3f}")

    summary = {
        "rows_analysed": int(len(df)),
        "total_sales_inr": round(total_sales, 2),
        "average_order_value_inr": round(avg_order, 2),
        "top_category": str(top_category),
        "top_region": str(top_region),
        "model_mae_inr": round(float(mae), 2),
        "model_rmse_inr": round(float(rmse), 2),
        "model_r2": round(float(r2), 4),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCharts and summary saved to: {OUT}")

if __name__ == "__main__":
    main()
