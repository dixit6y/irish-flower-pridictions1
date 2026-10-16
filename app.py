# This file runs the Flask website

from flask import Flask, render_template, request
from iris_model import predict_flower

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            # Get the numbers typed by the user
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Predict the flower
            result = predict_flower(sepal_length, sepal_width, petal_length, petal_width)
        except ValueError:
            error = "Please enter valid numbers in all boxes."

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
