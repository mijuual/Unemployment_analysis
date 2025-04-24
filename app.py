from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv; load_dotenv()
import os, openai
from core import load_and_predict, make_plot

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    country = request.form.get("country","Canada")  # default on first visit
    context = {"country": country, "error": None}

    try:
        df,future, preds, img_uri = load_and_predict(country)
        img_uri = make_plot(df, future, preds, country)   
        context.update({
            "years": df["Year"].tolist(),
            "rates": df["UnemploymentRate"].tolist(),
            "img_uri": img_uri
        })
    except ValueError as e:
        context["error"] = str(e)

    return render_template("index.html", **context)

# tiny JSON API the chatbot calls from browser JS
@app.route("/ask", methods=["POST"])
def ask():
    q = request.json["question"]
    ans = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role":"system","content":"You explain unemployment data."},
          {"role":"user","content":q}
        ]
    )["choices"][0]["message"]["content"]
    return jsonify(answer=ans)

if __name__ == "__main__":
    app.run(debug=True)
