import base64, io, os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

CSV = "SDG_0852_SEX_AGE_RT_A-filtered-2025-04-21 (1).csv"

def load_and_predict(country: str = "Canada"):
    df = (pd.read_csv(CSV)
            .rename(columns={
                "ref_area.label": "Country",
                "obs_value":      "UnemploymentRate",
                "time":           "Year"
            }))

    # Filter chosen country & years, drop NaNs
    df = (df[df["Country"].str.lower() == country.lower()]
            [["Year", "UnemploymentRate"]]
            .dropna())
    df["Year"] = df["Year"].astype(int)
    df = df.query("2015 <= Year <= 2025")

    X = df["Year"].values.reshape(-1, 1)
    y = df["UnemploymentRate"].values
    model = LinearRegression().fit(X, y)

    future = np.arange(2025, 2031).reshape(-1, 1)
    preds  = model.predict(future)

    return df, future.flatten().tolist(), preds.tolist(), model

def make_plot(df: pd.DataFrame,
              future_years: list[int],
              preds: list[float],
              country: str) -> str:
    
    X = df["Year"].values.reshape(-1, 1)
    y = df["UnemploymentRate"].values

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(X, y, label="Historical")
    ax.plot(X, LinearRegression().fit(X, y).predict(X), label="Fit")
    ax.plot(future_years, preds, "--", label="Forecast")

    ax.set(title=f"{country.title()} Unemployment",
           xlabel="Year", ylabel="Rate (%)")
    ax.legend()
    fig.tight_layout()

    # Encode PNG → base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)          # free memory when used in web server
    return f"data:image/png;base64,{img_b64}"