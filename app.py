from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    avg_area_income = float(request.form.get("avg_area_income"))
    avg_area_house_age = float(request.form.get("avg_area_house_age"))
    avg_area_rooms = float(request.form.get("avg_area_rooms"))
    avg_area_bedrooms = float(request.form.get("avg_area_bedrooms"))
    area_population = float(request.form.get("area_population"))

    # Predict the price
    features = np.array([[avg_area_income, avg_area_house_age, avg_area_rooms,
                          avg_area_bedrooms, area_population]])
    prediction = model.predict(features)
    price = round(prediction[0], 2)

    return jsonify({"price": price})

if __name__ == "__main__":
    app.run(debug=True)