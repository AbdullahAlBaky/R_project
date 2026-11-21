# Khulna Weather-Based Rainfall Prediction

A Flask web application that loads the saved Random Forest pipeline and predicts whether rain is expected on the following day using the same six input features used in the Google Colab notebook.

## Included
- `app.py` — Flask backend and prediction route
- `models/rainfall_prediction_model.pkl` — trained Random Forest pipeline
- `models/rainfall_features.pkl` — expected feature order
- `templates/index.html` — responsive dashboard
- `static/css/style.css` — visual styling
- `static/js/app.js` — date helper and form interaction

## Requirements
- Python 3.10–3.12 recommended
- VS Code (optional)
- Internet connection is not required to run the app after installing the Python packages.

> Important compatibility note: the model was saved with scikit-learn 1.6.1. The requirements file pins that version because loading scikit-learn pickle files across different versions is not guaranteed to work.

## Run on Windows (VS Code)

Open this folder in VS Code, then open Terminal and run:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

If PowerShell blocks activation, use Command Prompt in VS Code, or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Prediction inputs
Enter the observed day's:
- Rainfall (mm)
- Sunshine (hours)
- Humidity (%)
- Temperature (°C)
- Date (used to derive Month and DayOfYear)

The model predicts the next day's rainfall class. The probability shown is the model's predicted class probability, not a guaranteed weather forecast.

## Model evaluation results

The following values are the results recorded in the project's Colab evaluation:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 82.89% | 69.55% | 76.23% | 72.74% | 0.8886 | 0.7983 |
| Decision Tree | 80.32% | 70.00% | 60.00% | 64.62% | 0.7450 | 0.5398 |
| Random Forest | 85.42% | 84.91% | 62.40% | 71.94% | 0.9158 | 0.8358 |
| KNN | 82.99% | 77.16% | 61.37% | 68.36% | 0.8699 | 0.7340 |

Random Forest had the highest accuracy, precision, ROC-AUC, and Average Precision among the four evaluated models. Logistic Regression had the highest recall and F1-Score.

## Project limitations
- This is an educational ML demonstration, not an official weather service.
- Prediction quality depends on the training data and input quality.
- The probability is a model output and should not be treated as certainty.
- The model uses historical Khulna weather records and six input features.


## One-page project showcase
The website explains the dataset and target creation, data cleaning and chronological split, EDA findings, reasons for selecting the four classifiers, implementation workflow, prediction form, evaluation results, and project limitations.
