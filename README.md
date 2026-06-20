What the project produces (all working, tested):
The NAV report shows: AUM of ₹1.07 Cr, NAV per unit of ₹10.7673, 13.92% portfolio return, across 10 Indian equity holdings. Four charts plus an Excel workbook with 4 sheets.

What each section of the code teaches you:
create_sample_portfolio() — how to build and manipulate a pandas DataFrame, create derived columns, and model financial data. This mirrors exactly how Advent Geneva stores positions.
FundNAVCalculator class — your first Python class (OOP). The NAV formula (Assets − Liabilities) ÷ Units is what every fund calculates daily. This is the heart of Geneva's accounting engine.
simulate_historical_nav() — introduces Geometric Brownian Motion, the standard quant finance model. Uses numpy for random number generation and pandas for time series.
calculate_performance_metrics() — teaches Sharpe Ratio, drawdown, VaR, win rate — the exact metrics asked in every finance + data science interview.
create_all_charts() — matplotlib and seaborn for 4 different chart types: bar, pie, line with fill, and colour-coded P&L bars.
export_reports() — real-world skill: writing multi-sheet Excel files with openpyxl. Every data analyst role needs this.

file information :
requirments.txt - shows the tools and packges needed for this project.
nav_calculator.py - phython code for entire project that created 10 records.
rest all other files are output files created after running the code .excel outputs with png graphical representation of out put files.
