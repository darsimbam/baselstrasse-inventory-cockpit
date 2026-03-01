# Dagmersellen.py
# Dagmersellen Co. LTD – Inventory & Service Cockpit (Streamlit app)

import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

# ---- CONFIG ----
DATA_PATH = "supply_chain_data.csv"  # Update this path if your CSV is located elsewhere
HIGH_DOS = 70
LOW_DOS = 7
SLOW_MOVER_QUANTILE = 0.25
DEMAND_DAYS = 30.0


# ---------- 0. Load & standardize data ----------

@st.cache_data
def load_data(file=None, path: str = DATA_PATH) -> pd.DataFrame:
    """Load from uploaded file (if provided) or from default path."""
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(path)

    df = df.copy()
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

MIN_SLOW_DOS = 45  # SKU must also have high DOS to be flagged as slow mover

def analyze_dead_slow_and_dos(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["avg_daily_demand"] = (result["sales_qty"] / DEMAND_DAYS).replace(0, 0.01)
    result["days_of_supply"] = result["on_hand_qty"] / result["avg_daily_demand"]
    slow_threshold = result["sales_qty"].quantile(SLOW_MOVER_QUANTILE)
    result["slow_flag"] = (result["sales_qty"] <= slow_threshold) & (
        result["days_of_supply"] >= MIN_SLOW_DOS
    )
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
    """Uses 'order_qty' as a simple planned demand, 'sales_qty' as actual."""
    actual_qty = df["sales_qty"].replace(0, float("nan"))
    abs_error = (df["order_qty"] - df["sales_qty"]).abs()
    return pd.DataFrame({"ape": abs_error / actual_qty}, index=df.index)


# ---------- 4. AI Suggestions (rule-based "ML") ----------

def build_ai_suggestions(dos_df: pd.DataFrame) -> pd.DataFrame:
    sug = dos_df.copy()
    dos = sug["days_of_supply"]
    slow = sug["slow_flag"]

    # Vectorized policy: conditions checked in priority order (first match wins)
    conditions = [
        slow & (dos > HIGH_DOS),                   # Very slow + high DOS -> cut aggressively
        slow & (dos > 60),                          # Slow but not extreme -> moderate cut
        (dos < LOW_DOS) & (sug["sales_qty"] > 0),  # Low DOS, active demand -> protect
        dos > HIGH_DOS,                             # High DOS, not flagged slow -> reduce
    ]
    choices = [60.0, 45.0, 30.0, 90.0]
    sug["recommended_days_of_supply"] = np.select(conditions, choices, default=dos.values)

    sug["recommended_stock_qty"] = sug["recommended_days_of_supply"] * sug["avg_daily_demand"]
    sug["recommended_inventory_value"] = sug["recommended_stock_qty"] * sug["unit_price"]
    sug["inventory_value_delta"] = sug["recommended_inventory_value"] - sug["inventory_value"]

    sug["service_risk"] = pd.cut(
        dos,
        bins=[-np.inf, LOW_DOS, 15.0, np.inf],
        labels=["High (stockout risk)", "Medium", "Low"],
    )

    return sug


# ---------- 5. Planner Copilot (mock GenAI) ----------

def compute_kpis(dos_df: pd.DataFrame, fc_df: pd.DataFrame) -> dict:
    total_value = dos_df["inventory_value"].sum()
    slow_value = dos_df.loc[dos_df["slow_flag"], "inventory_value"].sum()
    return {
        "total_value": total_value,
        "slow_share": slow_value / total_value if total_value > 0 else 0.0,
        "high_dos_count": int((dos_df["days_of_supply"] > HIGH_DOS).sum()),
        "low_dos_count": int((dos_df["days_of_supply"] < LOW_DOS).sum()),
        "overall_turns": dos_df["sales_value"].sum() / total_value if total_value > 0 else 0.0,
        "mape_overall": fc_df["ape"].mean(skipna=True),
    }


_HELP_TEXT = (
    "I can answer questions about:\n"
    "- Executive summary / COO report\n"
    "- Dead stock and slow movers\n"
    "- Service risk and stockout exposure\n"
    "- Inventory turns and turnover by product type\n"
    "- Forecast accuracy (MAPE)\n"
    "- Urgent replenishment needs\n"
    "- Working capital tied up in inventory\n"
    "- Best selling / fast moving SKUs\n"
    "- Inventory breakdown by product type\n"
    "\nType 'help' to see this list again."
)


def planner_copilot_answer(
    question: str,
    dos_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    kpis: dict,
) -> str:
    q = (question or "").strip().lower()

    if not q:
        return "Please type a question. " + _HELP_TEXT

    slow_share = kpis["slow_share"]
    high_dos_count = kpis["high_dos_count"]
    low_dos_count = kpis["low_dos_count"]
    overall_turns = kpis["overall_turns"]
    mape_overall = kpis["mape_overall"]
    total_value = kpis["total_value"]

    # 1) Help / capabilities
    if any(k in q for k in ("help", "what can you", "what do you know", "capabilities", "what questions", "what topics")):
        return "Here is what I can help you with:\n\n" + _HELP_TEXT

    # 2) Executive summary / COO
    if any(k in q for k in ("summary", "coo", "executive", "overview", "report")):
        return (
            "Executive summary for Dagmersellen Co. LTD:\n\n"
            f"- Slow movers represent {slow_share:.1%} of total inventory value.\n"
            f"- {high_dos_count} SKUs have DOS > {HIGH_DOS} days, tying up working capital.\n"
            f"- {low_dos_count} SKUs have DOS < {LOW_DOS} days and are at stockout risk.\n"
            f"- Inventory turns: {overall_turns:.2f}x per period.\n"
            f"- Forecast MAPE: {mape_overall:.1%}.\n\n"
            "Takeaway: excess stock on slow movers while fast movers are under-protected. "
            "Use the AI Suggestions tab to identify specific SKUs to act on."
        )

    # 3) Dead stock / slow movers
    if any(k in q for k in ("dead stock", "dead", "slow mover", "slow stock", "slow", "obsolete")):
        slow_tbl = (
            dos_df[dos_df["slow_flag"]]
            .sort_values("inventory_value", ascending=False)
            [["sku", "product_type", "inventory_value", "days_of_supply"]]
            .head(5)
        )
        lines = [f"Slow movers represent {slow_share:.1%} of total inventory value. Top 5 by value:"]
        for r in slow_tbl.itertuples(index=False):
            lines.append(
                f"- {r.sku} ({r.product_type}): value {r.inventory_value:,.0f}, ~{r.days_of_supply:.0f} days of supply."
            )
        lines.append(
            "\nActions:\n"
            "- Freeze or reduce replenishment on these SKUs.\n"
            "- Plan markdowns or bundles for very high DOS items.\n"
            "- Review assortment for truly obsolete items and consider write-off."
        )
        return "\n".join(lines)

    # 4) Service risk / stockout / OTIF / fill rate
    if any(k in q for k in ("otif", "service", "stockout", "stock out", "out of stock", "fill rate", "risk")):
        low_tbl = (
            dos_df[dos_df["days_of_supply"] < LOW_DOS]
            .sort_values("days_of_supply")
            [["sku", "product_type", "days_of_supply", "sales_qty"]]
            .head(5)
        )
        if low_tbl.empty:
            return f"No SKUs are currently below the critical {LOW_DOS}-day DOS threshold. Service risk is low."
        lines = [f"{low_dos_count} SKU(s) are below {LOW_DOS} days of supply — at stockout risk:"]
        for r in low_tbl.itertuples(index=False):
            lines.append(
                f"- {r.sku} ({r.product_type}): {r.days_of_supply:.1f} days of supply, demand {r.sales_qty} units."
            )
        lines.append(
            "\nActions:\n"
            "- Raise urgent purchase orders for these SKUs.\n"
            "- Avoid running promotions on items already below safe DOS.\n"
            "- Explore substitute products if supplier lead time is long."
        )
        return "\n".join(lines)

    # 5) Inventory turns / turnover / rotation
    if any(k in q for k in ("turn", "turnover", "rotation", "velocity")):
        if turns_df.empty:
            return "No turnover data available for the current selection."
        lines = [f"Inventory turns by product type (overall: {overall_turns:.2f}x):"]
        for r in turns_df.itertuples(index=False):
            lines.append(
                f"- {r.product_type}: {r.turns:.2f}x  "
                f"(sales {r.sales_value:,.0f} / inventory {r.inventory_value:,.0f})"
            )
        best = turns_df.loc[turns_df["turns"].idxmax()]
        worst = turns_df.loc[turns_df["turns"].idxmin()]
        lines.append(
            f"\nBest: {best['product_type']} ({best['turns']:.2f}x)  |  "
            f"Worst: {worst['product_type']} ({worst['turns']:.2f}x)"
        )
        lines.append(
            "\nLow turns = capital tied up. Consider reducing safety stock "
            "or running promotions on low-turn categories."
        )
        return "\n".join(lines)

    # 6) Forecast accuracy / MAPE / error / prediction
    if any(k in q for k in ("forecast", "accuracy", "mape", "error", "predict", "demand plan")):
        threshold = 0.20
        assessment = (
            "within an acceptable range." if mape_overall <= threshold
            else f"above the {threshold:.0%} guideline — demand planning should be reviewed."
        )
        return (
            f"Forecast accuracy (prototype — using order qty as proxy for planned demand):\n\n"
            f"- Overall MAPE: {mape_overall:.1%} — {assessment}\n\n"
            "Improvement actions:\n"
            "- Collaborate with sales for forward-looking demand signals.\n"
            "- Apply statistical models (moving average, exponential smoothing).\n"
            "- Clean historical outliers (promotions, one-offs) before modelling.\n"
            "- Review SKU-level errors to identify systematic over- or under-forecasting."
        )

    # 7) Urgent replenishment / reorder / buy
    if any(k in q for k in ("replenish", "reorder", "order more", "urgent", "buy", "purchase", "restock")):
        urgent = (
            dos_df[dos_df["days_of_supply"] < LOW_DOS]
            .sort_values("days_of_supply")
            [["sku", "product_type", "days_of_supply", "avg_daily_demand", "on_hand_qty"]]
            .head(10)
        )
        if urgent.empty:
            return f"No SKUs are currently below the critical {LOW_DOS}-day threshold. No urgent replenishment required."
        lines = [f"Urgent replenishment required for {len(urgent)} SKU(s) below {LOW_DOS} days of supply:"]
        for r in urgent.itertuples(index=False):
            lines.append(
                f"- {r.sku} ({r.product_type}): {r.days_of_supply:.1f} days left, "
                f"avg demand {r.avg_daily_demand:.1f} units/day, on hand {r.on_hand_qty:.0f} units."
            )
        lines.append("\nAction: raise purchase orders immediately, prioritised by lowest DOS first.")
        return "\n".join(lines)

    # 8) Working capital / cash / tied up / liquidity
    if any(k in q for k in ("working capital", "capital", "cash", "tied up", "free up", "release", "liquidity", "money")):
        slow_capital = total_value * slow_share
        return (
            f"Working capital tied up in inventory: {total_value:,.0f}\n\n"
            f"- Slow movers account for {slow_share:.1%} of that ({slow_capital:,.0f}).\n"
            f"- {high_dos_count} SKUs have DOS > {HIGH_DOS} days, indicating excess stock.\n"
            f"- Current inventory turns: {overall_turns:.2f}x — higher turns = less capital required.\n\n"
            "To free up capital:\n"
            "- Reduce replenishment on slow/high-DOS SKUs (see AI Suggestions tab).\n"
            "- Negotiate consignment or return-to-vendor for obsolete items.\n"
            "- Apply markdown pricing to accelerate clearance of excess inventory."
        )

    # 9) Best sellers / fast movers / top SKUs
    if any(k in q for k in ("best seller", "fast mover", "top sku", "top product", "most sold", "highest demand", "best performing", "fast moving", "best")):
        top = (
            dos_df.sort_values("sales_qty", ascending=False)
            [["sku", "product_type", "sales_qty", "sales_value", "days_of_supply"]]
            .head(10)
        )
        lines = ["Top 10 SKUs by sales volume:"]
        for r in top.itertuples(index=False):
            dos_flag = " *** LOW STOCK" if r.days_of_supply < LOW_DOS else ""
            lines.append(
                f"- {r.sku} ({r.product_type}): {r.sales_qty:.0f} units, "
                f"value {r.sales_value:,.0f}, DOS {r.days_of_supply:.0f} days{dos_flag}"
            )
        lines.append(
            "\nEnsure these high-demand SKUs are well-stocked. "
            "Any flagged *** LOW STOCK need urgent replenishment."
        )
        return "\n".join(lines)

    # 10) Product type / category breakdown
    if any(k in q for k in ("product type", "category", "breakdown", "by type", "by category", "segment", "class", "split")):
        if turns_df.empty:
            return "No product type data available for the current selection."
        lines = [f"Inventory and sales breakdown by product type (total inventory: {total_value:,.0f}):"]
        for r in turns_df.itertuples(index=False):
            share = r.inventory_value / total_value if total_value > 0 else 0.0
            lines.append(
                f"- {r.product_type}: inventory {r.inventory_value:,.0f} ({share:.1%}), "
                f"sales {r.sales_value:,.0f}, turns {r.turns:.2f}x"
            )
        return "\n".join(lines)

    # No matching intent found
    return (
        "I don't have an answer for that question based on the available data.\n\n"
        + _HELP_TEXT
    )


# ---------- 6. Streamlit UI ----------

def main():
    st.set_page_config(
        page_title="Dagmersellen Inventory & Service Cockpit", layout="wide"
    )
    st.title("Dagmersellen Co. LTD – Inventory & Service Level Cockpit")

    st.sidebar.header("Controls")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

    try:
        df_raw = load_data(uploaded)
    except FileNotFoundError:
        st.error(
            "Could not find the default CSV.\n\n"
            f"Expected at: {DATA_PATH}\n"
            "Either upload a file in the sidebar or update DATA_PATH in the script."
        )
        st.stop()

    # Product filter
    product_types = ["All"] + sorted(df_raw["product_type"].unique())
    selected_type = st.sidebar.selectbox("Product type", product_types)
    if selected_type != "All":
        df = df_raw[df_raw["product_type"] == selected_type].copy()
    else:
        df = df_raw.copy()

    # Pre-compute analytics
    dos_df = analyze_dead_slow_and_dos(df)
    turns_df = analyze_inventory_turnover(df)
    fc_df = analyze_forecast_vs_actual(df)
    sug_df = build_ai_suggestions(dos_df)
    kpis = compute_kpis(dos_df, fc_df)

    # Header KPIs
    total_inv = df["inventory_value"].sum()
    total_skus = df["sku"].nunique()

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Product type", selected_type)
    c1.metric("Total inventory value", f"{total_inv:,.0f}")
    c2.metric("Total SKUs", total_skus)
    c3.metric("Overall turns", f"{kpis['overall_turns']:.2f}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["KPI Overview", "AI Suggestions (ML)", "Planner Copilot (GenAI)", "LinkedIn Carousel"]
    )

    # ---- TAB 1: KPI Overview ----
    with tab1:
        st.subheader("KPI Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slow-mover share of inventory value", f"{kpis['slow_share']:.1%}")
        c2.metric(f"SKUs with DOS > {HIGH_DOS} days", kpis["high_dos_count"])
        c3.metric(f"SKUs with DOS < {LOW_DOS} days", kpis["low_dos_count"])
        c4.metric("Overall MAPE", f"{kpis['mape_overall']:.1%}")

        cols_slow = ["sku", "product_type", "on_hand_qty", "sales_qty", "inventory_value", "days_of_supply"]
        cols_dos = ["sku", "product_type", "on_hand_qty", "sales_qty", "days_of_supply"]

        st.markdown("#### Top slow movers by inventory value")
        st.dataframe(
            dos_df[dos_df["slow_flag"]]
            .sort_values("inventory_value", ascending=False)[cols_slow],
            use_container_width=True,
        )

        st.markdown(f"#### High DOS (> {HIGH_DOS} days)")
        st.dataframe(
            dos_df[dos_df["days_of_supply"] > HIGH_DOS]
            .sort_values("days_of_supply", ascending=False)[cols_dos],
            use_container_width=True,
        )

        st.markdown(f"#### Low DOS (< {LOW_DOS} days)")
        st.dataframe(
            dos_df[dos_df["days_of_supply"] < LOW_DOS]
            .sort_values("days_of_supply")[cols_dos],
            use_container_width=True,
        )

        st.markdown("#### Inventory turns by product type")
        st.dataframe(turns_df, use_container_width=True)

    # ---- TAB 2: AI Suggestions ----
    with tab2:
        st.subheader("AI Suggestions for Inventory Optimization")

        total_delta = sug_df["inventory_value_delta"].sum()
        freed = -total_delta if total_delta < 0 else 0.0
        extra = total_delta if total_delta > 0 else 0.0

        c1, c2 = st.columns(2)
        c1.metric("Potential inventory reduction (if all suggestions applied)", f"{freed:,.0f}")
        c2.metric("Potential inventory increase to protect service", f"{extra:,.0f}")

        cols_sug = [
            "sku", "product_type", "days_of_supply", "recommended_days_of_supply",
            "inventory_value", "recommended_inventory_value", "inventory_value_delta", "service_risk",
        ]

        st.markdown("#### Top inventory reduction opportunities")
        st.dataframe(
            sug_df[sug_df["inventory_value_delta"] < 0]
            .sort_values("inventory_value_delta")[cols_sug]
            .head(20),
            use_container_width=True,
        )

        st.markdown("#### SKUs needing more protection (inventory increase)")
        st.dataframe(
            sug_df[sug_df["inventory_value_delta"] > 0]
            .sort_values("inventory_value_delta", ascending=False)[cols_sug]
            .head(20),
            use_container_width=True,
        )

    # ---- TAB 3: Planner Copilot ----
    with tab3:
        st.subheader("Planner Copilot – Ask questions and get instant insights")

        if "copilot_answer" not in st.session_state:
            st.session_state["copilot_answer"] = ""
        if "copilot_question" not in st.session_state:
            st.session_state["copilot_question"] = ""

        _QUICK_QUESTIONS = [
            ("Executive Summary",     "Give me an executive summary"),
            ("Dead Stock",            "How do we reduce dead stock?"),
            ("Service Risk",          "Where is service at risk?"),
            ("Inventory Turns",       "Show inventory turns by product type"),
            ("Forecast Accuracy",     "What is our forecast accuracy?"),
            ("Urgent Replenishment",  "Which SKUs need urgent replenishment?"),
            ("Working Capital",       "How much working capital is tied up?"),
            ("Best Sellers",          "Show me the best selling SKUs"),
            ("Category Breakdown",    "Give me a breakdown by product type"),
            ("Help",                  "help"),
        ]

        st.markdown("**Quick questions** — click to get an instant answer:")
        cols = st.columns(5)
        for i, (label, question) in enumerate(_QUICK_QUESTIONS):
            if cols[i % 5].button(label, key=f"qq_{i}", use_container_width=True):
                st.session_state["copilot_answer"] = planner_copilot_answer(
                    question, dos_df, turns_df, kpis
                )
                st.session_state["copilot_question"] = label

        st.divider()
        st.markdown("**Or ask your own question:**")
        custom_q = st.text_area(
            "", placeholder="e.g. 'Which products have the worst forecast error?'",
            height=80, label_visibility="collapsed",
        )
        if st.button("Ask", type="primary"):
            if custom_q.strip():
                st.session_state["copilot_answer"] = planner_copilot_answer(
                    custom_q, dos_df, turns_df, kpis
                )
                st.session_state["copilot_question"] = custom_q.strip()

        if st.session_state["copilot_answer"]:
            st.divider()
            if st.session_state["copilot_question"]:
                st.markdown(f"**Q: {st.session_state['copilot_question']}**")
            st.info(st.session_state["copilot_answer"])


    # ---- TAB 4: LinkedIn Carousel ----
    with tab4:
        st.subheader("Export as PDF for LinkedIn Carousel Post")
        st.caption("Square 1080×1080px slides — upload directly to LinkedIn as a document post.")

        if st.button("Generate Carousel PDF", type="primary"):
            pdf = FPDF(orientation="P", unit="mm", format=(102, 102))
            pdf.set_auto_page_break(False)

            BG      = (15,  23,  42)
            ACCENT  = (99, 179, 237)
            WHITE   = (255, 255, 255)
            MUTED   = (160, 174, 192)

            def add_slide(title, lines, badge=None):
                pdf.add_page()
                pdf.set_fill_color(*BG)
                pdf.rect(0, 0, 102, 102, "F")
                # accent bar
                pdf.set_fill_color(*ACCENT)
                pdf.rect(0, 0, 4, 102, "F")
                # badge (e.g. slide number)
                if badge:
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.set_text_color(*ACCENT)
                    pdf.set_xy(8, 6)
                    pdf.cell(0, 6, badge)
                # title
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(*WHITE)
                pdf.set_xy(8, 14)
                pdf.cell(86, 8, title)
                # divider
                pdf.set_draw_color(*ACCENT)
                pdf.set_line_width(0.4)
                pdf.line(8, 24, 94, 24)
                # body lines
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(*MUTED)
                y = 29
                for line in lines:
                    if line == "":
                        y += 3
                        continue
                    pdf.set_xy(8, y)
                    pdf.cell(86, 6, line)
                    y += 7
                # footer
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(*MUTED)
                pdf.set_xy(8, 93)
                pdf.cell(86, 5, "Dagmersellen Co. LTD  |  Inventory & Service Cockpit")

            # Slide 1 — Cover
            add_slide(
                "Dagmersellen Inventory Overview ",
                [
                    f"Total Inventory Value:  {total_inv:,.0f}",
                    f"Total SKUs:             {total_skus}",
                    f"Inventory Turns:        {kpis['overall_turns']:.2f}x",
                    f"Forecast MAPE:          {kpis['mape_overall']:.1%}",
                    f"Slow-mover share:       {kpis['slow_share']:.1%}",
                    f"SKUs with DOS > {HIGH_DOS}d:  {kpis['high_dos_count']}",
                    f"SKUs with DOS < {LOW_DOS}d:  {kpis['low_dos_count']}",
                ],
                badge="OVERVIEW",
            )

            # Slide 2 — Slow movers
            slow_top = (
                dos_df[dos_df["slow_flag"]]
                .sort_values("inventory_value", ascending=False)
                .head(5)
            )
            slow_lines = [f"Slow-mover share of inventory: {kpis['slow_share']:.1%}", ""]
            for r in slow_top.itertuples():
                slow_lines.append(f"- {r.sku} ({r.product_type}):  {r.inventory_value:,.0f}  |  {r.days_of_supply:.0f}d")
            add_slide("Slow Movers & Dead Stock", slow_lines, badge="RISK")

            # Slide 3 — Service risk
            low_dos = dos_df[dos_df["days_of_supply"] < LOW_DOS].sort_values("days_of_supply").head(5)
            risk_lines = [f"SKUs below {LOW_DOS}-day threshold: {kpis['low_dos_count']}", ""]
            if low_dos.empty:
                risk_lines.append("No critical stockouts detected.")
            else:
                for r in low_dos.itertuples():
                    risk_lines.append(f"- {r.sku}: {r.days_of_supply:.1f} days  -  demand {r.sales_qty} units")
            add_slide("Service Risk / Stockouts", risk_lines, badge="URGENT")

            # Slide 4 — Inventory turns by category
            turns_lines = [f"Overall turns: {kpis['overall_turns']:.2f}x", ""]
            for r in turns_df.itertuples():
                turns_lines.append(f"- {r.product_type}:  {r.turns:.2f}x")
            add_slide("Inventory Turns by Category", turns_lines, badge="TURNS")

            # Slide 5 — Top 5 best sellers
            top5 = dos_df.sort_values("sales_qty", ascending=False).head(5)
            top_lines = []
            for r in top5.itertuples():
                flag = "  [LOW]" if r.days_of_supply < LOW_DOS else ""
                top_lines.append(f"- {r.sku}: {r.sales_qty:.0f} units  |  DOS {r.days_of_supply:.0f}d{flag}")
            add_slide("Top 5 Best Sellers", top_lines, badge="TOP SKUs")

            # Slide 6 — Call to action
            add_slide(
                "Key Actions",
                [
                    f"1. Reduce stock on {kpis['high_dos_count']} high-DOS SKUs",
                    f"2. Urgently replenish {kpis['low_dos_count']} at-risk SKUs",
                    "3. Review slow movers for markdown / write-off",
                    "4. Improve forecast accuracy (current MAPE",
                    f"   {kpis['mape_overall']:.1%})",
                ],
                badge="ACTIONS",
            )

            pdf_bytes = bytes(pdf.output())
            st.download_button(
                label="Download Carousel PDF",
                data=pdf_bytes,
                file_name="linkedin_carousel.pdf",
                mime="application/pdf",
            )
            st.success("6 slides generated — upload the PDF as a LinkedIn Document post.")


if __name__ == "__main__":
    main()
