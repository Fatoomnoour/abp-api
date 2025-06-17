from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler
#from pyngrok import ngrok

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model and scalers
try:
    logger.info("Loading model and scalers...")
    model = load_model(
        "CNN_LSTM_Model_256.h5", custom_objects={"mse": MeanSquaredError()}
    )

    with open("scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open("scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    logger.info("Model and scalers loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scalers: {str(e)}")
    raise


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "API is running",
            "endpoints": {
                "/predict": {
                    "method": "POST",
                    "description": "Predict ABP values from PPG and ECG data",
                    "input_format": {
                        "ppg": "List of 250 PPG signal values",
                        "ecg": "List of 250 ECG signal values",
                    },
                }
            },
        }
    )


# Define a route for the prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from the client
        data = request.get_json()
        logger.info("Received request data")

        if not data:
            logger.error("No data provided in request")
            return jsonify({"error": "No data provided"}), 400

        if "ppg" not in data or "ecg" not in data:
            logger.error("Missing ppg or ecg data in request")
            return jsonify({"error": "Missing ppg or ecg data"}), 400

        # Assuming data contains 'ppg' and 'ecg' as lists
        ppg = np.array(data["ppg"])
        ecg = np.array(data["ecg"])
        logger.info(f"Input shapes - PPG: {ppg.shape}, ECG: {ecg.shape}")

        # Ensure the data is in the right shape (add any necessary preprocessing)
        sample_size = 250  # As in your model input size
        if len(ppg) != sample_size or len(ecg) != sample_size:
            print("error")
            logger.error(
                f"Invalid input size. Expected {sample_size}, got PPG: {len(ppg)}, ECG: {len(ecg)}"
            )
            return jsonify(
                {"error": f"Input data must be {sample_size} samples long"}
            ), 400

        ppg = ppg.reshape(1, sample_size)
        ecg = ecg.reshape(1, sample_size)

        # Stack PPG and ECG as the model expects 2 input channels
        X_input = np.stack((ppg, ecg), axis=-1)
        logger.info(f"Stacked input shape: {X_input.shape}")

        # Scale the input data
        X_scaled = scaler_X.transform(X_input.reshape(1, -1)).reshape(1, sample_size, 2)
        logger.info(f"Scaled input shape: {X_scaled.shape}")

        # Predict using the model
        prediction = model.predict(X_scaled)
        logger.info(f"Raw prediction shape: {prediction.shape}")

        # Inverse transform the prediction
        prediction_orig = scaler_y.inverse_transform(prediction)
        logger.info(f"Final prediction shape: {prediction_orig.shape}")

        return jsonify({"predicted_abp": prediction_orig.tolist()})

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting Flask server...")

# ngrok.set_auth_token("2wvZ8ttLx3FFlNtBzNOExpB9Idb_tivwrQiAR1qZoBjFXHty")

    #public_url = ngrok.connect(5000).public_url
   # logger.info(f' * ngrok tunnel "{public_url}" -> http://127.0.0.1:5000')
    #app.run(debug=False, host="0.0.0.0", port=5000)

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
