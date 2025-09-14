import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Previsor de Expectativa de Vida", layout="wide")

# =========================
# 1. Função Mock do Modelo
# =========================
def predict_life_expectancy_mock(data: pd.DataFrame) -> pd.Series:
    base_expectancy = 65.0
    schooling_effect = data["schooling"] * 0.8
    income_effect = data["income_composition_of_resources"] * 10
    mortality_effect = data["adult_mortality"] * (-0.05)
    hiv_effect = data["hiv_aids"] * -0.5

    prediction = base_expectancy + schooling_effect + income_effect + mortality_effect + hiv_effect
    return np.clip(prediction, 40, 95)  # valores entre 40 e 95


# =========================
# 2. Carregar Base de Dados
# =========================
DATA_URL = "https://raw.githubusercontent.com/bcmaymonegalvao/Life-Expectancy-Predicting-Model/main/Life%20Expectancy%20Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    # Normalizar nomes das colunas -> snake_case sem caracteres especiais
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df

input_df = load_data()

# =========================
# 3. Interface Streamlit
# =========================
st.title("🔮 Previsor de Expectativa de Vida")
st.write("Aplicação interativa que utiliza dados públicos para prever a expectativa de vida com base em fatores socioeconômicos e de saúde.")

# =========================
# 4. Filtros
# =========================
st.sidebar.header("Filtros")

countries = sorted(input_df["country"].unique())
years = sorted(input_df["year"].unique())

selected_countries = st.sidebar.multiselect("Selecione País(es)", options=countries, default=countries[:1])
selected_year = st.sidebar.selectbox("Selecione o Ano", options=["Todos"] + list(years))

# Escolha de variável para análise
available_vars = [c for c in input_df.columns if c not in ["country", "year", "status"]]
selected_var = st.sidebar.selectbox("Variável para análise de correlação", options=available_vars, index=available_vars.index("gdp"))

filtered_df = input_df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]
if selected_year != "Todos":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]

# Pré-processamento
processed_df = filtered_df.copy()
processed_df["status"] = processed_df["status"].map({"developing": 0, "developed": 1})

# =========================
# 5. Previsão
# =========================
if st.button("Prever Expectativa de Vida"):
    prediction = predict_life_expectancy_mock(processed_df)

    st.subheader("📊 Resultados da Previsão")
    st.write(processed_df.head())

    chart_data = pd.DataFrame({
        "Index": prediction.index,
        "Prediction": prediction.values
    })
    processed_df["prediction"] = prediction

    # --- Linha ---
    with st.container(border=True):
        st.markdown("### 📈 Linha")
        chart_line = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(x="Index", y="Prediction", tooltip=["Index", "Prediction"])
            .properties(height=300)
        )
        st.altair_chart(chart_line, use_container_width=True)

    # --- Histograma ---
    with st.container(border=True):
        st.markdown("### 📊 Histograma")
        chart_hist = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                alt.X("Prediction", bin=alt.Bin(maxbins=30), title="Expectativa de Vida"),
                alt.Y("count()", title="Frequência"),
                tooltip=["count()"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_hist, use_container_width=True)

    # --- Boxplot ---
    with st.container(border=True):
        st.markdown("### 📦 Boxplot")
        chart_box = (
            alt.Chart(chart_data)
            .mark_boxplot()
            .encode(y="Prediction")
            .properties(height=300)
        )
        st.altair_chart(chart_box, use_container_width=True)

    # --- Scatter (Predição vs variável escolhida) ---
    with st.container(border=True):
        st.markdown(f"### 🔎 Dispersão: Prediction vs {selected_var}")

        base = alt.Chart(processed_df).encode(
            x="prediction",
            y=selected_var,
            tooltip=["country", "year", "prediction", selected_var]
        )

        points = base.mark_circle(size=60, opacity=0.6, color="steelblue")
        regression_line = base.transform_regression("prediction", selected_var).mark_line(color="red")
        scatter = points + regression_line

        st.altair_chart(scatter.properties(height=400), use_container_width=True)

        # --- Cálculo da regressão (coeficientes) ---
        X = processed_df[["prediction"]].values
        y = processed_df[selected_var].values

        if len(X) > 1 and len(np.unique(y)) > 1:
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            r2 = model.score(X, y)

            st.markdown(
                f"📐 **Equação da Regressão:**  \n"
                f"y = {slope:.3f} × Prediction + {intercept:.3f}  \n"
                f"**R² = {r2:.3f}**"
            )
        else:
            st.warning("Não há variabilidade suficiente nos dados filtrados para calcular regressão.")

    # --- Heatmap de Correlação ---
    with st.container(border=True):
        st.markdown(f"### 🌡️ Heatmap de Correlação entre Prediction e {selected_var}")
        corr_df = processed_df[["prediction", selected_var]].corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", cbar=True, ax=ax)
        st.pyplot(fig)

