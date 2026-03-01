# beauty.py
# Streamlit prototype KPI app for the beauty distributor sample data.

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = r"C:\Users\simba\OneDrive\Dokumenter\data\archive\supply_chain_data.csv"
HIGH_DOS = 120
LOW_DOS = 5
SLOW_MOVER_QUANTILE = 0.2
DEMAND_DAYS = 30.0


# ---------- 0. Load & standardize data ----------

@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sku"] = df["SKU"]
    df["product_type"] = df["Product type"]
    df["on_hand_qty"] = df["Stock levels"]
    df["unit_price"] = df["Price"]
    df["sales_qty"] = df["Number of products sold"]
    df["order_qty"] = df["Order quantities"]
    df["inventory_value"] = df["on_hand_qty"] * df["unit_price"]
    df["sales_value"] = df["sales_qty"] * df["unit_price"]
    return df


# ---------- 1. Dead / Slow + Days of Supply ----------

def analyze_dead_slow_and_dos(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["avg_daily_demand"] = (result["sales_qty"] / DEMAND_DAYS).replace(0, 0.01)
    result["days_of_supply"] = result["on_hand_qty"] / result["avg_daily_demand"]
    slow_threshold = result["sales_qty"].quantile(SLOW_MOVER_QUANTILE)
    result["slow_flag"] = result["sales_qty"] <= slow_threshold
    return result


# ---------- 2. Inventory Turnover ----------

def analyze_inventory_turnover(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("product_type").agg(
        sales_value=("sales_value", "sum"),
        inventory_value=("inventory_value", "sum"),
    )
    grp["turns"] = (
        grp["sales_value"] / grp["inventory_value"].replace(0, float("nan"))
    ).fillna(0)
    return grp.reset_index()


# ---------- 3. Forecast vs Actual ----------

def analyze_forecast_vs_actual(df: pd.DataFrame) -> pd.DataFrame:
    """Uses 'order_qty' as planned demand, 'sales_qty' as actual."""
    fc = df.copy()
    fc["forecast_qty"] = fc["order_qty"]
    fc["actual_qty"] = fc["sales_qty"]
    fc["error"] = fc["forecast_qty"] - fc["actual_qty"]
    fc["abs_error"] = fc["error"].abs()
    fc["ape"] = fc["abs_error"] / fc["actual_qty"].replace(0, float("nan"))
    return fc


# ---------- 4. Streamlit UI ----------

def main():
    st.set_page_config(page_title="Beauty Supply KPIs", layout="wide")
    st.title("Beauty Distributor – Supply Chain KPIs (Prototype)")

    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    df = load_data(uploaded) if uploaded is not None else load_data()

    view = st.sidebar.radio(
        "Select view",
        ["Dead/Slow + Days of Supply", "Inventory Turnover", "Forecast vs Actual"],
    )
    st.sidebar.write(f"Rows: {len(df)}, SKUs: {df['sku'].nunique()}")

    if view == "Dead/Slow + Days of Supply":
        st.subheader("Dead/Slow Inventory & Days of Supply")
        dos_df = analyze_dead_slow_and_dos(df)

        total_value = dos_df["inventory_value"].sum()
        slow_value = dos_df.loc[dos_df["slow_flag"], "inventory_value"].sum()
        slow_share = slow_value / total_value if total_value > 0 else 0.0
        high_dos_count = (dos_df["days_of_supply"] > HIGH_DOS).sum()
        low_dos_count = (dos_df["days_of_supply"] < LOW_DOS).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Slow-mover share of inventory value", f"{slow_share:.1%}")
        c2.metric(f"SKUs with DOS > {HIGH_DOS} days", int(high_dos_count))
        c3.metric(f"SKUs with DOS < {LOW_DOS} days", int(low_dos_count))

        cols_slow = ["sku", "product_type", "on_hand_qty", "sales_qty", "inventory_value", "days_of_supply"]
        cols_dos = ["sku", "product_type", "on_hand_qty", "sales_qty", "days_of_supply"]

        st.markdown("#### Top slow movers by inventory value")
        st.dataframe(
            dos_df[dos_df["slow_flag"]].sort_values("inventory_value", ascending=False)[cols_slow],
            use_container_width=True,
        )

        st.markdown(f"#### High DOS (> {HIGH_DOS} days)")
        st.dataframe(
            dos_df[dos_df["days_of_supply"] > HIGH_DOS].sort_values("days_of_supply", ascending=False)[cols_dos],
            use_container_width=True,
        )

        st.markdown(f"#### Low DOS (< {LOW_DOS} days)")
        st.dataframe(
            dos_df[dos_df["days_of_supply"] < LOW_DOS].sort_values("days_of_supply")[cols_dos],
            use_container_width=True,
        )

    elif view == "Inventory Turnover":
        st.subheader("Inventory Turnover")
        turns_df = analyze_inventory_turnover(df)

        overall_inv = df["inventory_value"].sum()
        overall_turns = df["sales_value"].sum() / overall_inv if overall_inv > 0 else 0.0

        st.metric("Overall (simplified) inventory turns", f"{overall_turns:.2f}")
        st.markdown("#### Turns by product type")
        st.dataframe(turns_df, use_container_width=True)

    else:  # Forecast vs Actual
        st.subheader("Forecast vs Actual")
        fc_df = analyze_forecast_vs_actual(df)

        st.metric("Overall MAPE", f"{fc_df['ape'].mean(skipna=True):.1%}")

        st.markdown("#### MAPE by product type")
        st.dataframe(
            fc_df.groupby("product_type")["ape"].mean().reset_index().rename(columns={"ape": "mape"}),
            use_container_width=True,
        )

        st.markdown("#### SKU-level view")
        st.dataframe(
            fc_df[["sku", "product_type", "forecast_qty", "actual_qty", "error", "abs_error", "ape"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
