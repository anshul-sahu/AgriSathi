from flask import Flask, render_template, request, jsonify
from fertilizer_recomm import predict_fertilizer
from crop_pred import crop_prediction
import numpy as np

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route('/crop_recommendation_page')
def crop_recom_page():
    return render_template('crop_recommendation.html')

@app.route('/disease_prediction_page')
def disease_prediction():
    return render_template('disease_pred.html')

@app.route('/predict_crop', methods=["POST"])
def predict_crop():
    data = request.get_json()
    result = crop_prediction(
        int(data['n']), int(data['p']), int(data['k']),
        float(data['temp']), float(data['humidity']), float(data['ph']), float(data['rainfall'])
    )

    if isinstance(result, np.ndarray):
        result = result[0]  

    return jsonify({"crop": str(result)})

@app.route("/fertilizer_prediction_page")
def fert_pred_page():
    return render_template('fertilizer_pred.html')

@app.route("/predict_fertilizer", methods=["POST"])
def predict():
    data = request.get_json()

    result = predict_fertilizer(
        int(data["temp"]),
        int(data["humidity"]),
        int(data["moisture"]),
        data["soil_type"],
        data["crop_type"],
        int(data["n"]),
        int(data["p"]),
        int(data["k"])
    )

    return jsonify({"fertilizer": result}) 

app.run(debug=True)