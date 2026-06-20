"""
=============================================================
 FUND NAV CALCULATOR  — Data Science Portfolio Project
 Author  : Prashant Patil
 Domain  : Finance / Investment Management
 Tools   : Python, pandas, numpy, matplotlib, seaborn
 Purpose : Calculate Net Asset Value (NAV) for a mutual fund /
           hedge fund portfolio — the same concept used inside
           Advent Geneva every day.
=============================================================

WHAT IS NAV?
   NAV = (Total Assets - Total Liabilities) / Units Outstanding
   Every fund (mutual fund, hedge fund, ETF) calculates this
   daily so investors know the per-unit price of the fund.

WHAT THIS PROJECT COVERS:
   1. Portfolio data modelling with pandas DataFrames
   2. NAV calculation logic step by step
   3. Performance metrics (returns, drawdown, Sharpe ratio)
   4. Data visualisation with matplotlib & seaborn
   5. Export to CSV & Excel for reporting
   6. Clean, well-documented Python code for your portfolio
"""

# ─────────────────────────────────────────────
#  IMPORTS  (standard beginner data science stack)
# ─────────────────────────────────────────────
import pandas as pd          # data manipulation  — your primary tool
import numpy as np           # numerical computing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  1.  SAMPLE PORTFOLIO DATA
#      In real life this comes from Advent Geneva /
#      a database / an API.  Here we create it manually
#      so you can run the project without any external data.
# ─────────────────────────────────────────────

def create_sample_portfolio():
    """
    Build a realistic fund portfolio with 10 holdings.

    Returns
    -------
    pd.DataFrame  — one row per holding
    """
    holdings = {
        "security_id":   ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK",
                          "WIPRO",    "HUL", "LT",   "AXISBANK", "BAJFINANCE"],

        "security_name": ["Reliance Industries", "Tata Consultancy Services",
                          "Infosys Ltd",         "HDFC Bank",
                          "ICICI Bank",          "Wipro Ltd",
                          "Hindustan Unilever",  "Larsen & Toubro",
                          "Axis Bank",           "Bajaj Finance"],

        "asset_class":   ["Equity", "Equity", "Equity", "Equity", "Equity",
                          "Equity", "Equity", "Equity", "Equity", "Equity"],

        "sector":        ["Energy",  "Technology", "Technology", "Finance", "Finance",
                          "Technology", "FMCG",    "Infrastructure", "Finance", "Finance"],

        "quantity":      [500, 300, 800, 400, 1000,
                          600, 700, 250, 900, 200],

        # Current market price per share (₹)
        "market_price":  [2450.50, 3720.00, 1580.25, 1640.75, 980.30,
                          480.60,  2700.00, 3100.00,  1030.45, 6800.00],

        # Price at which we originally bought the stock (₹)
        "cost_price":    [2100.00, 3200.00, 1400.00, 1500.00, 850.00,
                          420.00,  2400.00, 2800.00,  900.00, 5900.00],

        # Accrued income (dividends declared but not yet paid)
        "accrued_income": [5000, 3000, 8000, 4000, 2000,
                           1500, 7000, 2500,  500, 1000],
    }

    df = pd.DataFrame(holdings)

    # ── Derived columns (calculated from the raw data above) ──
    # Market value = quantity × market price
    df["market_value"] = df["quantity"] * df["market_price"]

    # Cost value = quantity × cost price
    df["cost_value"] = df["quantity"] * df["cost_price"]

    # Unrealised P&L = market value − cost value
    df["unrealised_pnl"] = df["market_value"] - df["cost_value"]

    # Return % per holding
    df["return_pct"] = ((df["market_price"] - df["cost_price"])
                        / df["cost_price"] * 100).round(2)

    return df


# ─────────────────────────────────────────────
#  2.  NAV CALCULATION ENGINE
# ─────────────────────────────────────────────

class FundNAVCalculator:
    """
    Core NAV calculation engine for an equity fund.

    Think of this class as a simplified version of what
    Advent Geneva does under the hood every day.

    Parameters
    ----------
    fund_name        : str   — display name of the fund
    units_outstanding: float — total investor units in the fund
    liabilities      : float — fund expenses, fees payable (₹)
    cash_balance     : float — uninvested cash held by fund (₹)
    """

    def __init__(self, fund_name: str,
                 units_outstanding: float,
                 liabilities: float = 0.0,
                 cash_balance: float = 0.0):

        self.fund_name         = fund_name
        self.units_outstanding = units_outstanding
        self.liabilities       = liabilities
        self.cash_balance      = cash_balance
        self.portfolio         = None   # will be set by load_portfolio()

    # ── Step 1: Load the portfolio ──
    def load_portfolio(self, df: pd.DataFrame):
        """Attach the holdings DataFrame to this calculator."""
        self.portfolio = df.copy()
        print(f"✅  Portfolio loaded: {len(df)} holdings")
        return self

    # ── Step 2: Calculate NAV ──
    def calculate_nav(self) -> dict:
        """
        Core NAV formula:
            NAV per unit = (Total Assets - Total Liabilities)
                           ÷ Units Outstanding

        Total Assets = Market Value of Securities
                     + Accrued Income
                     + Cash Balance

        Returns a dictionary of all key fund metrics.
        """
        if self.portfolio is None:
            raise ValueError("No portfolio loaded. Call load_portfolio() first.")

        # ── Sum up all asset values ──
        total_market_value  = self.portfolio["market_value"].sum()
        total_accrued_income = self.portfolio["accrued_income"].sum()
        total_cost_value    = self.portfolio["cost_value"].sum()
        total_unrealised_pnl = self.portfolio["unrealised_pnl"].sum()

        # Total Assets
        total_assets = total_market_value + total_accrued_income + self.cash_balance

        # Total Liabilities (expenses payable, management fees, etc.)
        total_liabilities = self.liabilities

        # Net Assets = Assets − Liabilities
        net_assets = total_assets - total_liabilities

        # NAV per Unit (the number printed on every fund statement)
        nav_per_unit = net_assets / self.units_outstanding

        # Fund-level return %
        total_return_pct = (total_unrealised_pnl / total_cost_value) * 100

        # Store results in a clean dictionary
        results = {
            "fund_name"          : self.fund_name,
            "calculation_date"   : datetime.today().strftime("%Y-%m-%d"),
            "total_market_value" : round(total_market_value, 2),
            "total_accrued_income": round(total_accrued_income, 2),
            "cash_balance"       : round(self.cash_balance, 2),
            "total_assets"       : round(total_assets, 2),
            "total_liabilities"  : round(total_liabilities, 2),
            "net_assets"         : round(net_assets, 2),
            "units_outstanding"  : self.units_outstanding,
            "nav_per_unit"       : round(nav_per_unit, 4),
            "total_cost_value"   : round(total_cost_value, 2),
            "total_unrealised_pnl": round(total_unrealised_pnl, 2),
            "total_return_pct"   : round(total_return_pct, 2),
        }

        self.nav_results = results
        return results

    # ── Step 3: Print a clean NAV summary ──
    def print_nav_summary(self):
        """Print a formatted fund summary — like a fund fact sheet."""
        r = self.nav_results
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  FUND NAV REPORT — {r['fund_name']}")
        print(f"  Date: {r['calculation_date']}")
        print(sep)
        print(f"  {'Market Value of Securities':<30} ₹{r['total_market_value']:>15,.2f}")
        print(f"  {'Accrued Income':<30} ₹{r['total_accrued_income']:>15,.2f}")
        print(f"  {'Cash Balance':<30} ₹{r['cash_balance']:>15,.2f}")
        print(f"  {'-'*47}")
        print(f"  {'Total Assets':<30} ₹{r['total_assets']:>15,.2f}")
        print(f"  {'Total Liabilities':<30} ₹{r['total_liabilities']:>15,.2f}")
        print(f"  {'-'*47}")
        print(f"  {'Net Assets (AUM)':<30} ₹{r['net_assets']:>15,.2f}")
        print(f"  {'Units Outstanding':<30}  {r['units_outstanding']:>15,.0f}")
        print(f"  {'-'*47}")
        print(f"  {'★ NAV per Unit':<30} ₹{r['nav_per_unit']:>15,.4f}")
        print(f"  {'Total Unrealised P&L':<30} ₹{r['total_unrealised_pnl']:>15,.2f}")
        print(f"  {'Total Return %':<30}  {r['total_return_pct']:>14.2f}%")
        print(f"{sep}\n")


# ─────────────────────────────────────────────
#  3.  HISTORICAL NAV SIMULATION
#      Simulate 1 year of daily NAV history so we can
#      analyse fund performance over time.
# ─────────────────────────────────────────────

def simulate_historical_nav(base_nav: float,
                             days: int = 252,
                             annual_return: float = 0.14,
                             volatility: float = 0.18,
                             seed: int = 42) -> pd.DataFrame:
    """
    Simulate realistic daily NAV values using Geometric Brownian Motion —
    the standard model used in quantitative finance.

    Parameters
    ----------
    base_nav       : float — starting NAV value
    days           : int   — number of trading days (252 = 1 year)
    annual_return  : float — expected yearly return (0.14 = 14%)
    volatility     : float — annual volatility / risk (0.18 = 18%)
    seed           : int   — random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: date, nav, daily_return, cumulative_return
    """
    np.random.seed(seed)

    # Convert annual parameters to daily
    daily_return = annual_return / 252
    daily_vol    = volatility / np.sqrt(252)

    # Generate random daily returns (normally distributed)
    random_shocks = np.random.normal(daily_return, daily_vol, days)

    # Build NAV series: each day's NAV = previous × (1 + return)
    nav_values = [base_nav]
    for shock in random_shocks:
        nav_values.append(nav_values[-1] * (1 + shock))

    # Create date range (business days only, skipping weekends)
    start_date = datetime.today() - timedelta(days=int(days * 1.4))
    dates = pd.bdate_range(start=start_date, periods=days + 1)

    df = pd.DataFrame({
        "date": dates,
        "nav" : nav_values,
    })

    # Daily return %
    df["daily_return"]      = df["nav"].pct_change() * 100
    # Cumulative return from day 1
    df["cumulative_return"] = ((df["nav"] / df["nav"].iloc[0]) - 1) * 100

    # Rolling 30-day max (for drawdown calculation)
    df["rolling_max"] = df["nav"].cummax()
    # Drawdown = how far we are from the peak (negative number)
    df["drawdown"]    = ((df["nav"] - df["rolling_max"]) / df["rolling_max"]) * 100

    return df


# ─────────────────────────────────────────────
#  4.  PERFORMANCE METRICS
# ─────────────────────────────────────────────

def calculate_performance_metrics(hist_df: pd.DataFrame,
                                  risk_free_rate: float = 0.065) -> dict:
    """
    Calculate key fund performance metrics used by analysts.

    Parameters
    ----------
    hist_df        : pd.DataFrame — output of simulate_historical_nav()
    risk_free_rate : float — annual risk-free rate (Indian T-bill ≈ 6.5%)

    Returns
    -------
    dict of metrics
    """
    returns = hist_df["daily_return"].dropna() / 100   # as decimals

    # ── Annualised Return ──
    total_return = (hist_df["nav"].iloc[-1] / hist_df["nav"].iloc[0]) - 1
    n_years      = len(hist_df) / 252
    annualised_return = (1 + total_return) ** (1 / n_years) - 1

    # ── Annualised Volatility (risk) ──
    annualised_vol = returns.std() * np.sqrt(252)

    # ── Sharpe Ratio: (Return - Risk Free) / Volatility ──
    # Higher Sharpe = better risk-adjusted return. >1 is good, >2 is great.
    sharpe_ratio = (annualised_return - risk_free_rate) / annualised_vol

    # ── Maximum Drawdown: worst peak-to-trough loss ──
    max_drawdown = hist_df["drawdown"].min()

    # ── Win Rate: % of days with positive return ──
    win_rate = (returns > 0).sum() / len(returns) * 100

    # ── VaR (Value at Risk) at 95% confidence ──
    # "On 95% of days, we don't lose more than this %"
    var_95 = np.percentile(returns, 5) * 100

    metrics = {
        "Total Return %"         : round(total_return * 100, 2),
        "Annualised Return %"    : round(annualised_return * 100, 2),
        "Annualised Volatility %" : round(annualised_vol * 100, 2),
        "Sharpe Ratio"           : round(sharpe_ratio, 2),
        "Max Drawdown %"         : round(max_drawdown, 2),
        "Win Rate %"             : round(win_rate, 2),
        "VaR 95% (Daily) %"     : round(var_95, 2),
        "Best Day %"             : round(returns.max() * 100, 2),
        "Worst Day %"            : round(returns.min() * 100, 2),
    }

    return metrics


# ─────────────────────────────────────────────
#  5.  VISUALISATIONS
# ─────────────────────────────────────────────

def create_all_charts(portfolio_df: pd.DataFrame,
                      hist_df: pd.DataFrame,
                      nav_results: dict,
                      output_dir: str = "outputs"):
    """
    Generate 4 charts and save them as PNG files.
    All charts together form a professional fund dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Chart style ──
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
              "#3B1F2B", "#44BBA4", "#E94F37", "#393E41",
              "#F5A623", "#7B2D8B"]

    # ═══════════════════════════════════════════
    #  Chart 1: Portfolio Holdings — Market Value Bar Chart
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{nav_results['fund_name']} — Portfolio Overview",
                 fontsize=15, fontweight="bold", y=1.01)

    # Left: market value by stock
    ax1 = axes[0]
    bars = ax1.barh(portfolio_df["security_id"],
                    portfolio_df["market_value"] / 1e5,   # in Lakhs
                    color=colors[:len(portfolio_df)])
    ax1.set_xlabel("Market Value (₹ Lakhs)", fontsize=11)
    ax1.set_title("Market Value per Holding", fontsize=12, fontweight="bold")

    # Add value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.5, bar.get_y() + bar.get_height()/2,
                 f"₹{w:.1f}L", va="center", fontsize=9)

    # Right: sector allocation pie
    ax2 = axes[1]
    sector_data = portfolio_df.groupby("sector")["market_value"].sum()
    wedges, texts, autotexts = ax2.pie(
        sector_data, labels=sector_data.index,
        autopct="%1.1f%%", colors=colors,
        startangle=90, pctdistance=0.8
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax2.set_title("Sector Allocation", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path1 = os.path.join(output_dir, "1_portfolio_overview.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Chart 1 saved: {path1}")

    # ═══════════════════════════════════════════
    #  Chart 2: Historical NAV Line Chart
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist_df["date"], hist_df["nav"],
            color="#2E86AB", linewidth=1.8, label="NAV")
    ax.fill_between(hist_df["date"], hist_df["nav"],
                    hist_df["nav"].min(), alpha=0.12, color="#2E86AB")

    # Highlight max and min points
    max_idx = hist_df["nav"].idxmax()
    min_idx = hist_df["nav"].idxmin()
    ax.scatter(hist_df.loc[max_idx, "date"], hist_df.loc[max_idx, "nav"],
               color="#27AE60", s=80, zorder=5, label=f"Peak ₹{hist_df.loc[max_idx,'nav']:.2f}")
    ax.scatter(hist_df.loc[min_idx, "date"], hist_df.loc[min_idx, "nav"],
               color="#C73E1D", s=80, zorder=5, label=f"Trough ₹{hist_df.loc[min_idx,'nav']:.2f}")

    ax.set_title(f"{nav_results['fund_name']} — Historical NAV (1 Year)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("NAV per Unit (₹)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path2 = os.path.join(output_dir, "2_historical_nav.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Chart 2 saved: {path2}")

    # ═══════════════════════════════════════════
    #  Chart 3: Drawdown Chart
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(hist_df["date"], hist_df["drawdown"], 0,
                    color="#C73E1D", alpha=0.55, label="Drawdown")
    ax.plot(hist_df["date"], hist_df["drawdown"],
            color="#C73E1D", linewidth=0.8)
    ax.set_title("Drawdown from Peak — Risk Visualisation",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path3 = os.path.join(output_dir, "3_drawdown.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Chart 3 saved: {path3}")

    # ═══════════════════════════════════════════
    #  Chart 4: P&L per Holding — Bar Chart
    # ═══════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ["#27AE60" if v >= 0 else "#C73E1D"
                  for v in portfolio_df["unrealised_pnl"]]
    bars = ax.bar(portfolio_df["security_id"],
                  portfolio_df["unrealised_pnl"] / 1000,   # in '000s
                  color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Unrealised P&L per Holding (₹ '000s)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Security", fontsize=11)
    ax.set_ylabel("P&L (₹ '000s)", fontsize=11)
    plt.xticks(rotation=20)

    # Add return % labels on bars
    for bar, row in zip(bars, portfolio_df.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (2 if h >= 0 else -8),
                f"{row.return_pct:+.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path4 = os.path.join(output_dir, "4_pnl_per_holding.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅  Chart 4 saved: {path4}")


# ─────────────────────────────────────────────
#  6.  EXPORT TO CSV & EXCEL
# ─────────────────────────────────────────────

def export_reports(portfolio_df: pd.DataFrame,
                   hist_df: pd.DataFrame,
                   nav_results: dict,
                   metrics: dict,
                   output_dir: str = "outputs"):
    """Save data to CSV and Excel files for reporting."""
    os.makedirs(output_dir, exist_ok=True)

    # ── CSV 1: Portfolio holdings ──
    portfolio_df.to_csv(os.path.join(output_dir, "portfolio_holdings.csv"),
                        index=False)

    # ── CSV 2: Historical NAV ──
    hist_df.to_csv(os.path.join(output_dir, "historical_nav.csv"), index=False)

    # ── Excel: All sheets in one workbook ──
    excel_path = os.path.join(output_dir, "fund_nav_report.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        # Sheet 1 — Holdings
        portfolio_df.to_excel(writer, sheet_name="Holdings", index=False)

        # Sheet 2 — NAV Summary
        summary_df = pd.DataFrame(
            list(nav_results.items()), columns=["Metric", "Value"]
        )
        summary_df.to_excel(writer, sheet_name="NAV Summary", index=False)

        # Sheet 3 — Performance Metrics
        metrics_df = pd.DataFrame(
            list(metrics.items()), columns=["Metric", "Value"]
        )
        metrics_df.to_excel(writer, sheet_name="Performance", index=False)

        # Sheet 4 — Historical NAV
        hist_df[["date", "nav", "daily_return",
                 "cumulative_return", "drawdown"]].to_excel(
            writer, sheet_name="Historical NAV", index=False
        )

    print(f"✅  Excel report saved: {excel_path}")
    print(f"✅  CSV files saved to: {output_dir}/")


# ─────────────────────────────────────────────
#  7.  MAIN — Run everything end to end
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  FUND NAV CALCULATOR — Running Full Analysis")
    print("=" * 55)

    # ── Step 1: Build portfolio ──
    print("\n[1/5] Building sample portfolio...")
    portfolio = create_sample_portfolio()
    print(portfolio[["security_id", "quantity", "market_price",
                      "market_value", "return_pct"]].to_string(index=False))

    # ── Step 2: Calculate NAV ──
    print("\n[2/5] Calculating NAV...")
    calculator = FundNAVCalculator(
        fund_name         = "India Growth Equity Fund",
        units_outstanding = 1_000_000,        # 10 lakh units
        liabilities       = 250_000,          # ₹2.5 lakh in expenses
        cash_balance      = 500_000,          # ₹5 lakh cash held
    )
    calculator.load_portfolio(portfolio)
    nav_results = calculator.calculate_nav()
    calculator.print_nav_summary()

    # ── Step 3: Simulate historical NAV ──
    print("[3/5] Simulating 1 year of NAV history...")
    hist_df = simulate_historical_nav(
        base_nav       = nav_results["nav_per_unit"] * 0.86,  # start 14% lower
        days           = 252,
        annual_return  = 0.14,
        volatility     = 0.18,
    )
    # Force last NAV to match our calculated NAV
    hist_df.loc[hist_df.index[-1], "nav"] = nav_results["nav_per_unit"]

    # ── Step 4: Performance metrics ──
    print("[4/5] Calculating performance metrics...")
    metrics = calculate_performance_metrics(hist_df)
    print("\n  ── PERFORMANCE METRICS ──")
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")

    # ── Step 5: Charts & Reports ──
    print("\n[5/5] Generating charts and reports...")
    create_all_charts(portfolio, hist_df, nav_results, output_dir="outputs")
    export_reports(portfolio, hist_df, nav_results, metrics, output_dir="outputs")

    print("\n✅  All done! Check the 'outputs/' folder.")
    print("=" * 55 + "\n")

    return portfolio, hist_df, nav_results, metrics


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
