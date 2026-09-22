"""
Week 4 Homework — Gold Price Time Series (DATS 6401)

Run locally:  streamlit run app.py
Needs:        pip install -r requirements.txt
The CSV file must be in the same folder as this file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="Gold Price Time Series", layout="centered")

BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
plt.rcParams.update({"axes.grid": True, "grid.color": "#e8e7e3",
                     "axes.spines.top": False, "axes.spines.right": False})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# DATA
# =====================================================================
@st.cache_data
def load_daily():
    df = pd.read_csv("gold_data_daily_comprehensive_cleaned.csv", usecols=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    failed = int(df["Date"].isna().sum())
    gold = df.dropna(subset=["Date"]).set_index("Date")["Close"].sort_index()
    return gold, failed


def resample(gold, rule, min_days):
    """Average the daily prices in each period; keep only complete periods."""
    average = gold.resample(rule).mean()
    days = gold.resample(rule).count()
    return average[days >= min_days], days


gold, failed = load_daily()
monthly, days_in_month = resample(gold, "MS", 15)
add = seasonal_decompose(monthly, period=12, model="additive")
mult = seasonal_decompose(monthly, period=12, model="multiplicative")

# =====================================================================
# SIDEBAR — resolution widget
# =====================================================================
RESOLUTION = {"Weekly": ("W", 3, 52), "Monthly": ("MS", 15, 12), "Quarterly": ("QS", 40, 4)}
UNIT = {"Weekly": "week", "Monthly": "month", "Quarterly": "quarter"}

st.sidebar.header("Resolution")
res = st.sidebar.radio("Resolution for Graphs 1, 2 and 6:", list(RESOLUTION), index=1)
rule, min_days, per_year = RESOLUTION[res]
series, _ = resample(gold, rule, min_days)
st.sidebar.write(f"**{len(series)}** {res.lower()} values")
st.sidebar.caption("Monthly is the resolution used in this homework. "
                   "Steps 5–6 always use monthly data, because they study the 12-month season.")

# =====================================================================
# TITLE
# =====================================================================
st.title("Gold Price Time Series (2000–2026)")
st.markdown("""
**DATS 6401 — Visualization of Complex Data · Week 4 Homework**

**Data:** daily gold close price (Yahoo Finance), 30 Aug 2000 – 6 Feb 2026, 6,383 trading days.
**Resolution:** one average close price per month.
""")

# =====================================================================
# STEP 1
# =====================================================================
st.header("Step 1 — Load the data and convert dates")
c1, c2, c3 = st.columns(3)
c1.metric("Trading days", f"{len(gold):,}")
c2.metric("Failed dates", failed)
c3.metric("Duplicate dates", int(gold.index.duplicated().sum()))
st.code('df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")\n'
        'gold = df.set_index("Date")["Close"].sort_index()', language="python")
st.markdown(f"""
**Explanation.** Dates in a CSV file are text. `pd.to_datetime` turns them into real dates.
`format` stops pandas from guessing, and `errors="coerce"` marks bad dates as `NaT`.
Result: **{failed}** failed dates, no duplicates, rows sorted by time.
""")

# =====================================================================
# STEP 2
# =====================================================================
st.header("Step 2 — Resample: one average price per month")
st.code('monthly_all   = gold.resample("MS").mean()\n'
        'days_in_month = gold.resample("MS").count()\n'
        'monthly = monthly_all[days_in_month >= 15]', language="python")
st.table(pd.DataFrame({
    "Month": [days_in_month.index[0].strftime("%b %Y"), days_in_month.index[-1].strftime("%b %Y")],
    "Trading days": [int(days_in_month.iloc[0]), int(days_in_month.iloc[-1])],
    "Decision": ["removed", "removed"],
}).set_index("Month"))
st.markdown(f"""
**Explanation.** `resample("MS")` groups the daily prices by month, and `.mean()` takes the average.
The first and last months have only 2 and 5 trading days, so they were removed.
Result: **{len(monthly)} complete months** ({monthly.index.min():%b %Y} – {monthly.index.max():%b %Y}), no gaps.
""")

# =====================================================================
# STEP 3 — Graph 1
# =====================================================================
st.header("Step 3 — First look: linear axis vs. log axis")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, log in zip(axes, [False, True]):
    ax.plot(series, color=BLUE)
    if log:
        ax.set_yscale("log")
    ax.set_title("Log axis" if log else "Linear axis")
    ax.set_ylabel("USD per ounce" + (" (log scale)" if log else ""))
fig.suptitle(f"Graph 1 — Gold, {res.lower()} average close price", fontweight="bold")
plt.tight_layout()
show(fig)

years = (series.index[-1] - series.index[0]).days / 365.25
growth = series.iloc[-1] / series.iloc[0]
st.markdown(f"""
**Graph 1 — explanation.** Same data, two axes. The price grew **{growth:.0f} times**
(USD {series.iloc[0]:,.0f} → USD {series.iloc[-1]:,.0f}, about +{(growth ** (1 / years) - 1) * 100:.1f}% per year).
On the linear axis the early years look flat. On the log axis, equal **percentage** changes have equal height.

**Conclusion:** we use the log axis for the trend.
""")

# =====================================================================
# STEP 4 — Graph 2
# =====================================================================
st.header("Step 4 — Trend: 1-year rolling mean")
trend = series.rolling(per_year, center=True).mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series, color=GREY, linewidth=1, label=f"{res.lower()} average")
ax.plot(trend, color=BLUE, linewidth=2.5, label=f"{per_year}-{UNIT[res]} centred rolling mean")
ax.set_yscale("log")
ax.set_ylabel("USD per ounce (log scale)")
ax.set_title("Graph 2 — Trend: centred 1-year rolling mean", fontweight="bold")
ax.legend()
show(fig)
st.markdown(f"""
**Graph 2 — explanation.** Grey = {res.lower()} data. Blue = average of one year ({per_year} values).
A one-year average removes short movements and shows the **trend**.
`center=True` keeps the line in the middle of its window, not shifted to the right.

**Conclusion:** growth 2001–2011, flat 2012–2018, fast growth after 2019.
""")

# =====================================================================
# STEP 5 — Graph 3
# =====================================================================
st.header("Step 5 — Checking for seasonality: autocorrelation (ACF)")
monthly_return = np.log(monthly).diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
plot_acf(monthly, lags=36, ax=axes[0], color=BLUE, vlines_kwargs={"colors": BLUE})
axes[0].set_title("Price level")
plot_acf(monthly_return, lags=36, ax=axes[1], color=ORANGE, vlines_kwargs={"colors": ORANGE})
axes[1].set_title("Monthly % change (trend removed)")
for ax in axes:
    ax.set_xlabel("lag (months)")
fig.suptitle("Graph 3 — Autocorrelation (ACF), monthly data", fontweight="bold")
plt.tight_layout()
show(fig)
lag12 = acf(monthly_return, nlags=12)[12]
st.markdown(f"""
**Graph 3 — explanation.** The ACF compares the series with itself *k* months earlier.
The shaded area means "no real correlation".
- **Left:** the trend makes every lag high, so we cannot see seasons.
- **Right:** with the trend removed, lag 12 is only **{lag12:.2f}** (CO₂ in the lecture: 0.92).

**Conclusion:** gold has no strong yearly cycle.
""")

# =====================================================================
# STEP 6 — Graphs 4 and 5
# =====================================================================
st.header("Step 6 — Decomposition: trend, seasonality and noise")
st.markdown("""
- **Additive:** price = trend **+** season **+** noise (season = fixed number of dollars)
- **Multiplicative:** price = trend **×** season **×** noise (season = fixed percentage)

We choose the model whose noise (residual) keeps the **same size** over time.
""")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(add.resid, color=ORANGE, linewidth=0.9)
axes[0].axhline(0, color="#bbbbbb")
axes[0].set_title("Additive residual (USD)")
axes[1].plot((mult.resid - 1) * 100, color=GREEN, linewidth=0.9)
axes[1].axhline(0, color="#bbbbbb")
axes[1].set_title("Multiplicative residual (%)")
fig.suptitle("Graph 4 — Which model fits? Compare the residuals", fontweight="bold")
plt.tight_layout()
show(fig)

blocks = [("2001", "2005"), ("2011", "2015"), ("2021", "2026")]
st.table(pd.DataFrame({
    "Period": [f"{a}–{b}" for a, b in blocks],
    "Additive residual (± USD)": [round(add.resid[a:b].std(), 1) for a, b in blocks],
    "Multiplicative residual (± %)": [round(mult.resid[a:b].std() * 100, 1) for a, b in blocks],
}).set_index("Period"))
st.markdown("""
**Graph 4 — explanation.** Additive residuals grow with the price (±17 → ±65 USD).
Multiplicative residuals stay at about ±3–4% for all 25 years.

**Conclusion:** we use the **multiplicative** model.
""")

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(monthly, color=GREY);                                 axes[0].set_ylabel("Observed (USD)")
axes[1].plot(mult.trend, color=BLUE, linewidth=2);                 axes[1].set_ylabel("Trend (USD)")
axes[2].plot((mult.seasonal - 1) * 100, color=GREEN);              axes[2].set_ylabel("Seasonal (%)")
axes[3].plot((mult.resid - 1) * 100, color=ORANGE, linewidth=0.8); axes[3].set_ylabel("Residual (%)")
axes[3].axhline(0, color="#bbbbbb")
axes[0].set_yscale("log"); axes[1].set_yscale("log")
axes[0].set_title("Graph 5 — Multiplicative decomposition (period = 12 months)", fontweight="bold")
plt.tight_layout()
show(fig)
st.markdown("""
**Graph 5 — explanation.** The price is split into **trend**, **seasonal** and **residual** (noise).
The seasonal part moves only between −1.2% and +1.4%. The residual often moves ±5%.

**Conclusion:** the noise is bigger than the season, so gold's seasonality is **weak**.
""")

# =====================================================================
# STEP 7 — Uncertainty (Graph 6)
# =====================================================================
st.header("Step 7 — Uncertainty: a band around the trend")
roll = series.rolling(per_year, center=True)
mean, std = roll.mean(), roll.std()
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, color=BLUE, alpha=0.2, linewidth=0,
                label=f"±2 standard deviations of the {per_year} values in the window")
ax.plot(series, color=GREY, linewidth=0.8, label=f"{res.lower()} average")
ax.plot(mean, color=BLUE, linewidth=2, label="1-year rolling mean")
ax.set_yscale("log")
ax.set_ylabel("USD per ounce (log scale)")
ax.set_title("Graph 6 — Trend with a ±2 standard deviation band", fontweight="bold")
ax.legend(loc="upper left")
show(fig)
st.code("roll = series.rolling(12, center=True)\n"
        "mean, std = roll.mean(), roll.std()\n"
        "ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.2)", language="python")
st.markdown(f"""
**Graph 6 — explanation.** A single line looks more certain than the data really is.
The shaded band is **mean ± 2 standard deviations** of the prices in each 1-year window,
so it contains about **95% of the prices**. The band is **wide** when gold moved fast
(2006, 2011, 2024–2025) and **narrow** in calm years (2014–2018).

**Note:** the band shows the **spread of the data**, not the precision of the trend line.

**Conclusion:** uncertainty is not constant — it grows when the market is volatile.
""")

# =====================================================================
# STEP 8 — Graph 7
# =====================================================================
st.header("Step 8 — Resolution: daily, monthly, quarterly")
daily_part = gold["2015":"2026"]
monthly_part = monthly["2015":"2026"]
quarterly_part = monthly.resample("QS").mean()["2015":"2026"]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=True)
for ax, (s_, title, c) in zip(axes, [(daily_part, "Daily", GREY),
                                     (monthly_part, "Monthly average", BLUE),
                                     (quarterly_part, "Quarterly average", GREEN)]):
    ax.plot(s_.index, s_.values, color=c, linewidth=1.4)
    ax.set_title(f"{title} (n = {len(s_)})")
axes[0].set_ylabel("USD per ounce")
fig.suptitle("Graph 7 — The same period (2015–2026) at three resolutions", fontweight="bold")
plt.tight_layout()
show(fig)
st.markdown(f"""
**Graph 7 — explanation.** More averaging gives a smoother line but **hides extremes**:
highest daily price **USD {gold.max():,.0f}**, highest monthly average **USD {monthly.max():,.0f}**.
Monthly fits our question (trend and yearly season, 12 points per year).
Use the sidebar widget to see Graphs 1, 2 and 6 at other resolutions.
""")

# =====================================================================
# STEP 9 — Honesty note
# =====================================================================
st.header("Step 9 — Note on temporal honesty")
st.markdown(f"""
1. **Y-axis.** The price grew 17 times, so trend charts use a **log axis**. No axis is truncated.
2. **Resolution.** Monthly averages fit the question, but they hide daily extremes
   (USD {gold.max():,.0f} daily vs. USD {monthly.max():,.0f} monthly).
3. **Incomplete months.** Aug 2000 (2 days) and Feb 2026 (5 days) were removed.
4. **Smoothing.** The trend uses a centred 12-month window, stated in the legend.
5. **Uncertainty.** The ±2 standard deviation band is labelled as the spread of the data, not the precision of the trend.
6. **Honest result.** The seasonal effect is smaller than the noise, so seasonality is reported as **weak**.
""")

st.header("Summary")
st.table(pd.DataFrame({
    "Question": ["Date handling", "Resampling", "Trend", "Seasonality", "Uncertainty"],
    "Answer": [f"{len(gold):,} dates converted, {failed} errors, sorted",
               f"{len(monthly)} complete months; 2 incomplete months removed",
               "Strong upward trend: 17 times in 25 years, +11.9% per year",
               f"Weak: ACF lag 12 = {lag12:.2f}; seasonal effect smaller than noise",
               "±2 standard deviation band around the trend (Graph 6)"],
}).set_index("Question"))

st.caption("DATS 6401 · Week 4 homework · Data: Yahoo Finance daily gold close price")