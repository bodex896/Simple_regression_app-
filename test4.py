from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Create a sample dataset
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) / 1.5

# Calculate coefficients using ordinary least squares
X_mean = np.mean(X)
y_mean = np.mean(y)
X_dev = X - X_mean
y_dev = y - y_mean
beta1 = np.sum(X_dev * y_dev) / np.sum(X_dev ** 2)
beta0 = y_mean - beta1 * X_mean

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        x = float(request.form["x"])

        # Make prediction
        y_pred = beta0 + beta1 * x

        # Render result
        return render_template("result.html", y_pred=y_pred)
    return render_template("index.html")

@app.route("/visualize")
def visualize():
    # Render plot (using HTML table instead of image)
    return render_template("visualize.html", X=X.tolist(), y=y.tolist())

if __name__ == "__main__":
    app.run(debug=True)
