from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        text = request.form["news"]
        text_vec = vectorizer.transform([text])
        result = model.predict(text_vec)[0]

        prediction = "Real News" if result == 0 else "Fake News"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
