import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Simple HTML interface
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Naive Bayes Classifier</title>
</head>
<body>
    <h2>Naive Bayes Model Trainer</h2>
    <form action="/train" method="post" enctype="multipart/form-data">
        <label>Upload CSV:</label>
        <input type="file" name="file" required><br><br>

        <label>Target Column Name:</label>
        <input type="text" name="target" required><br><br>

        <button type="submit">Train Model</button>
    </form>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html_page)


@app.route("/train", methods=["POST"])
def train_model():

    file = request.files['file']
    target_column = request.form['target']

    df = pd.read_csv(file)

    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = X.fillna(X.mode().iloc[0])

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return f"""
    <h2>Model Results</h2>
    <p><b>Accuracy:</b> {accuracy*100:.2f}%</p>

    <h3>Confusion Matrix</h3>
    <pre>{cm}</pre>

    <h3>Classification Report</h3>
    <pre>{report}</pre>

    <br><br>
    <a href="/">Train another dataset</a>
    """

if __name__ == "__main__":
    app.run()
