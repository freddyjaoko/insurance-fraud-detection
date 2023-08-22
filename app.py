import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        uploaded_file = request.files['csv_file']
        if not uploaded_file:
            return "No file uploaded", 400

        data = pd.read_csv(uploaded_file)

        # Preprocessing
        cat_df = data.select_dtypes(include=['object'])
        cat_df = pd.get_dummies(cat_df, drop_first=True)
        
        num_df = data.select_dtypes(include=['int'])

        X = pd.concat([num_df, cat_df], axis=1)

        # Prediction
        samples_to_predict = np.array(X)
        predictions = classifier.predict(samples_to_predict)

        return render_template("prediction_result.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
