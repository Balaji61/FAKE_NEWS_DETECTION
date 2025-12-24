from flask import Flask, render_template, request
from model import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        news_text = request.form["news"]
        result = predict_news(news_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
