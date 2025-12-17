from flask import Flask, render_template, request
import joblib
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "final_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            gender = request.form["gender"]
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            duration = float(request.form["duration"])
            heart_rate = float(request.form["heart_rate"])
            body_temp = float(request.form["body_temp"])

            # Encode gender same as training
            gender = 0 if gender == "male" else 1

            input_df = pd.DataFrame([{
                "Age": age,
                "Gender": gender,
                "Height": height,
                "Weight": weight,
                "Duration": duration,
                "Heart_Rate": heart_rate,
                "Body_Temp": body_temp
            }])

            prediction = round(model.predict(input_df)[0], 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
