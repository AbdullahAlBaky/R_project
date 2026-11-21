from datetime import date, datetime
from pathlib import Path
import warnings

import joblib
import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rainfall_prediction_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "rainfall_features.pkl"

app = Flask(__name__)

# The model was saved from scikit-learn 1.6.1 in Google Colab.
# Keep requirements.txt pinned to that version for best compatibility.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

MODEL_RESULTS = [
    {"name": "Logistic Regression", "accuracy": 82.89, "precision": 69.55, "recall": 76.23, "f1": 72.74, "auc": 0.8886, "ap": 0.7983},
    {"name": "Decision Tree", "accuracy": 80.32, "precision": 70.00, "recall": 60.00, "f1": 64.62, "auc": 0.7450, "ap": 0.5398},
    {"name": "Random Forest", "accuracy": 85.42, "precision": 84.91, "recall": 62.40, "f1": 71.94, "auc": 0.9158, "ap": 0.8358},
    {"name": "KNN", "accuracy": 82.99, "precision": 77.16, "recall": 61.37, "f1": 68.36, "auc": 0.8699, "ap": 0.7340},
]

RF_IMPORTANCE = [
    {"name": "Humidity", "value": 0.255720},
    {"name": "Sunshine", "value": 0.209668},
    {"name": "DayOfYear", "value": 0.174681},
    {"name": "Temperature", "value": 0.168391},
    {"name": "Rainfall", "value": 0.138245},
    {"name": "Month", "value": 0.053296},
]

def parse_float(field_name, label, minimum, maximum):
    raw = request.form.get(field_name, "").strip()
    if not raw:
        raise ValueError(f"Please enter {label.lower()}.")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_values = {
        "date": request.form.get("date", date.today().isoformat()),
        "rainfall": request.form.get("rainfall", ""),
        "sunshine": request.form.get("sunshine", ""),
        "humidity": request.form.get("humidity", ""),
        "temperature": request.form.get("temperature", ""),
    }

    if request.method == "POST":
        try:
            selected_date = datetime.strptime(form_values["date"], "%Y-%m-%d").date()
            rainfall = parse_float("rainfall", "Rainfall", 0, 1000)
            sunshine = parse_float("sunshine", "Sunshine", 0, 24)
            humidity = parse_float("humidity", "Humidity", 0, 100)
            temperature = parse_float("temperature", "Temperature", -10, 60)

            values = {
                "Rainfall": rainfall,
                "Sunshine": sunshine,
                "Humidity": humidity,
                "Temperature": temperature,
                "Month": selected_date.month,
                "DayOfYear": selected_date.timetuple().tm_yday,
            }
            input_array = np.array([[values[name] for name in features]], dtype=float)
            predicted_class = int(model.predict(input_array)[0])
            probabilities = model.predict_proba(input_array)[0]
            rain_probability = float(probabilities[1]) * 100
            no_rain_probability = float(probabilities[0]) * 100

            prediction = {
                "label": "Rain Expected Tomorrow" if predicted_class == 1 else "No Rain Expected Tomorrow",
                "is_rain": predicted_class == 1,
                "rain_probability": round(rain_probability, 1),
                "no_rain_probability": round(no_rain_probability, 1),
                "input_date": selected_date.strftime("%d %B %Y"),
                "tomorrow_date": (selected_date.fromordinal(selected_date.toordinal() + 1)).strftime("%d %B %Y"),
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction could not be completed. Please check your inputs and try again."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_values=form_values,
        results=MODEL_RESULTS,
        rf_importance=RF_IMPORTANCE,
    )

if __name__ == "__main__":
    app.run(debug=True)
