from flask import Flask, render_template, request
from fraudsecurity.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

CATEGORIES = [
    "food_dining", "gas_transport", "shopping_pos", "shopping_net",
    "entertainment", "health_fitness", "misc_net", "misc_pos",
    "kids_pets", "home", "personal_care", "travel",
    "grocery_net", "grocery_pos"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = CustomData(
            category=request.form["category"],
            amt=float(request.form["amt"]),
            gender=request.form["gender"],
            city_pop=int(request.form["city_pop"]),
            dob=request.form["dob"],
            hour=int(request.form["hour"]),
            is_weekend=request.form["is_weekend"],
            merchant=request.form["merchant"],
            zip_code=request.form["zip"]
        )

        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)[0]

        result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"

        return render_template("result.html", result=result)

    return render_template("index.html", categories=CATEGORIES)


if __name__ == "__main__":
    app.run(debug=True)
