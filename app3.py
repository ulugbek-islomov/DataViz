# app3.py Homework_1

import streamlit as st
import plotly.express as px

st.set_page_config(page_title = "Gapminder Explorer", layout = "wide")
st.title("Gapminder: Wealth and Healt of Worldwide Nations")

# Loading data

@st.cache_data
def load_data():
    return px.data.gapminder()

df = load_data()


# Preview
st.subheader("Data preview")
st.write(f"Shape:  {df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Column types"):
    st.dataframe(df.dtypes.astype(str).rename("dtype"))

# Chart

st.subheader("Life expectancy versus GDP per person")
year = st.slider("Year", 1952, 2007, 2007, step=5)
d = df[df.year ==year]

fig = px.scatter(d, x="gdpPercap", y="lifeExp", size = "pop", color = "continent", hover_name="country", log_x=True,
                size_max=60, range_x = [200, 60000], range_y = [25, 90], 
                labels={
                    "gdpPercap": "GDP per person", "lifeExp":"Life expectancy (years)","continent":"Continent", "pop":"Population", 
                },
                title = f"Life expectancy versus GDP per capita, {year}",)
fig.update_layout(legend_title_text="Continent")

st.plotly_chart(fig, use_container_width=True)
st.caption("Source: Gapminder Foundation, through plotly.express.data.gapminder(). "
            "Marker area is proportional to population.")

# Description 
st.markdown("""
**Dataset.** Gapminder is dataset included into Plotly Express library. It has 1,704 observations covering 142 countries in 5 year intervals from 1952 to 2007. The unit of observation is the country and year. The data is tidy — one row per country per year, one variable per column.

**Variables.** `country` and `continent` are categorical; `year` is temporal; `lifeExp` (years), `pop` (population) and `gdpPercap`are measures.

**Chart.** Life expectancy against GDP per capita for the selected year. GDP is described on a logarithmic axis because income is heavily right-skewed; on a linear axis the poorest 100 countries collapse against the left edge. Marker area encodes population and colour encodes continent, so four variables are legible
at once. Axis ranges are fixed across years so that movement between years shows real change rather than rescaling.

**What it shows.** The relationship between income and life expectancy is more concave than linear. Under approximately \\$5,000 per capita, small increase in GDP correspond to large gains in life expectancy, but above approximately \\$10,000, the curve flattens and additional income shows little further improvement. Continents tend to group into clusters: African countries concentrated in the lower-left region and European countries in the upper-right. Moving the year slider from 1952 to 2007 shifts the entire distribution upward and to the right, indicating that both income and life expectancy increased globally over the period.
""")