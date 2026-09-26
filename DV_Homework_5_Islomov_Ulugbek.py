# Week 5 homework: Gold Price Explorer
# Turns the Week 4 gold time-series analysis into an interactive Streamlit app.
# Key idea: Streamlit re-runs this whole file top to bottom on every click,
# so slow steps are cached and every widget simply returns its current value.

# Imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(page_title="Gold Price Explorer", layout="wide")
st.title("Gold Price Explorer")
st.caption("Daily gold price (USD per ounce), 2000 to today. Week 4 time-series analysis, now interactive.")

# Lookup tables
RULES = {"Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
PERIODS_PER_YEAR = {"Weekly": 52, "Monthly": 12, "Quarterly": 4}


# Loading the  data
# @st.cache_data saves the result, so the download runs once instead of on every click.
# ttl="1d" refreshes the saved copy once a day. If Yahoo returns nothing, the app falls back to gold.csv stored in the repo.
@st.cache_data(ttl="1d")
def load_gold():
    raw = yf.download("GC=F", start="2000-01-01", progress=False, auto_adjust=False)
    if raw.empty:
        return pd.read_csv("gold.csv", index_col="Date", parse_dates=True)["Close"]
    s = raw["Close"].squeeze().dropna()
    s.name = "Close"
    s.index.name = "Date"
    return s


# Trend analysis
# Week 4 steps: resample daily prices to weekly/monthly/quarterly averages,
# then add a centred rolling mean (trend) and a ±2 standard-deviation band.
@st.cache_data
def trend_table(daily, resolution, years):
    price = daily.resample(RULES[resolution]).mean()
    window = years * PERIODS_PER_YEAR[resolution]
    roll = price.rolling(window, center=True, min_periods=window // 2)
    return pd.DataFrame({
        "price": price,
        "trend": roll.mean(),
        "lower": roll.mean() - 2 * roll.std(),
        "upper": roll.mean() + 2 * roll.std(),
    })


daily = load_gold()

# Sidebar controls (widgets)
# Each widget returns the user's current choice as a normal Python value.
st.sidebar.header("Controls")
first, last = daily.index.year.min(), daily.index.year.max()
start, end = st.sidebar.slider("Years", first, last, (first, last), key="years")
resolution = st.sidebar.radio("Resolution", list(RULES), index=1, key="resolution")
window = st.sidebar.slider("Trend window (years)", 1, 5, 1, key="window")
log_scale = st.sidebar.toggle("Log scale", value=False, key="log")
show_band = st.sidebar.checkbox("Show ±2 sd band", value=True, key="band")

# Collapsible "About" box: explains the data and the method without cluttering the page.
with st.sidebar.expander("About"):
    st.write(
        "Data: Yahoo Finance gold futures (GC=F). The trend is a centred rolling mean; "
        "the band is ±2 standard deviations around it. On a log scale, equal "
        "vertical steps mean equal percentage changes."
    )

# Applying the user's choices
table = trend_table(daily, resolution, window)
view = table.loc[str(start):str(end)]

# Headline numbers (layout: 4 columns)
p0, p1 = view["price"].iloc[0], view["price"].iloc[-1]
n_years = (view.index[-1] - view.index[0]).days / 365.25
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest average price", f"${p1:,.0f}")
c2.metric("Growth", f"×{p1 / p0:.1f}")
c3.metric("Average growth per year", f"{((p1 / p0) ** (1 / n_years) - 1) * 100:.1f}%" if n_years > 0 else "–")
c4.metric("Highest daily close", f"${daily.loc[str(start):str(end)].max():,.0f}")

# Tabs (layout: three views on one page)
tab_trend, tab_season, tab_data = st.tabs(["Trend", "Seasonality", "Data"])

# Trend tab: price, trend line and ±2 sd band
with tab_trend:
    fig = go.Figure()
    if show_band:
        fig.add_trace(go.Scatter(x=view.index, y=view["upper"], line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=view.index, y=view["lower"], line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(42,120,214,0.15)",
                                 name="±2 sd band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=view.index, y=view["price"], name=f"{resolution} average",
                             line=dict(color="#2a78d6", width=1.5)))
    fig.add_trace(go.Scatter(x=view.index, y=view["trend"], name=f"{window}-year trend",
                             line=dict(color="#eb6834", width=2.5)))
    fig.update_layout(yaxis_title="USD per ounce", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40))
    if log_scale:   # log scale: equal vertical steps = equal % changes (Week 4)
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")
    st.caption("Wider trend windows smooth more. The band widens as prices rise, "
               "which is why a log scale (multiplicative view) suits gold better.")

# Seasonality tab: is gold stronger in some months?
with tab_season:
    monthly = daily.loc[str(start):str(end)].resample("ME").mean()
    change = monthly.pct_change().dropna() * 100
    by_month = change.groupby(change.index.month).mean() - change.mean()
    by_month.index = pd.to_datetime(by_month.index, format="%m").strftime("%b")  # 1 -> "Jan"
    fig2 = px.bar(x=by_month.index, y=by_month.values,
                  labels={"x": "Month", "y": "Difference from an average month (%)"})
    fig2.update_traces(marker_color="#2a78d6", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    st.plotly_chart(fig2, width="stretch")
    st.caption(f"Compared with an average month, calendar months range from {by_month.min():.1f}% to {by_month.max():.1f}%, "
               f"but a typical month moves ±{change.std():.1f}%. The seasonal effect is small "
               "compared with the noise, so seasonality in gold is weak.")

# Data tab: table and CSV download
with tab_data:
    st.dataframe(view.round(2), width="stretch")
    st.download_button("Download this view (CSV)", view.round(2).to_csv(),
                       file_name=f"gold_{resolution.lower()}_{start}-{end}.csv", mime="text/csv")