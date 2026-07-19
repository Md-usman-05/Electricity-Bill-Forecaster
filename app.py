from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/electricity_model.pkl")


def calculate_bill(units):
    """
    Simple electricity tariff calculation.
    """

    if units <= 100:
        return units * 3.5

    elif units <= 200:
        return 350 + (units - 100) * 5

    elif units <= 300:
        return 850 + (units - 200) * 6.5

    else:
        return 1500 + (units - 300) * 8


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    family = int(request.form["family"])

    ac = int(request.form["ac"])

    fans = int(request.form["fans"])

    fridge = int(request.form["fridge"])

    tv = int(request.form["tv"])

    washing = int(request.form["washing"])

    hours = int(request.form["hours"])

    previous = int(request.form["previous"])

    temperature = int(request.form["temperature"])

    sample = pd.DataFrame({

        "Family_Members":[family],

        "AC_Count":[ac],

        "Fan_Count":[fans],

        "Refrigerator_Count":[fridge],

        "TV_Count":[tv],

        "WashingMachine_UsesPerWeek":[washing],

        "Daily_Usage_Hours":[hours],

        "Previous_Month_Units":[previous],

        "Average_Temperature_C":[temperature]

    })

    predicted_units = model.predict(sample)[0]

    predicted_bill = calculate_bill(predicted_units)

    return render_template(

        "index.html",

        units=round(predicted_units,2),

        bill=round(predicted_bill,2)

    )


if __name__ == "__main__":
    app.run(debug=True)