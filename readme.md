# 🏥 Medical Cost Predictor

A full-stack Machine Learning web app that predicts annual medical insurance charges based on patient details. Built with Python, Scikit-learn, Flask, and deployed on Render.com.

🔗 **Live Demo:** [medical-cost-predictor-b95i.onrender.com](https://medical-cost-predictor-b95i.onrender.com)  
📁 **GitHub:** [github.com/Kshitij-Mangal/medical-cost-predictor](https://github.com/Kshitij-Mangal/medical-cost-predictor)

---

## 📸 Preview

> A user fills in their age, BMI, smoker status, region etc. → clicks Predict → instantly sees the estimated insurance charge + a comparison chart of all 6 ML models.

---

## 📁 Project Structure

```
medical-cost-predictor/
│
├── app.py                  # Flask backend (routes, model loading, prediction)
├── ml.py                   # ML training script (run once to generate .pkl files)
├── insurance.csv           # Dataset
│
├── lr.pkl                  # Trained Linear Regression model
├── ridge.pkl               # Trained Ridge model
├── dt.pkl                  # Trained Decision Tree model
├── rf.pkl                  # Trained Random Forest model
├── gb.pkl                  # Trained Gradient Boosting model
├── knn.pkl                 # Trained KNN model
├── scaler.pkl              # Trained StandardScaler
│
├── templates/
│   └── index.html          # Frontend UI (form + live prediction + chart)
│
├── requirements.txt        # Python dependencies
├── Procfile                # Render/gunicorn start command
├── render.yaml             # Render deployment config
├── runtime.txt             # Python version pin (3.10.5)
└── README.md               # This file
```

---

## 📊 Dataset

**File:** `insurance.csv`  
**Source:** Standard insurance dataset (Kaggle)

| Column | Type | Description | Encoding |
|--------|------|-------------|----------|
| `age` | int | Age of patient | Raw number |
| `sex` | int | Gender | male=1, female=0 |
| `bmi` | float | Body Mass Index | Raw number |
| `children` | int | Number of children | Raw number |
| `smoker` | int | Smoking status | yes=1, no=0 |
| `region` | int | US region | southwest=1, southeast=0, others=3 |
| `charges` | float | Annual insurance cost (**target**) | Raw number |

---

## 🤖 Machine Learning (`ml.py`)

### What it does:
1. Loads `insurance.csv` using pandas
2. Splits into train/test using `train_test_split`
3. Scales features using `StandardScaler` — normalizes all values to same scale
4. Trains 6 ML models in a loop
5. Evaluates each using R² score
6. Saves all models + scaler as `.pkl` files using pickle

### Models trained:

| Model | File | Description |
|-------|------|-------------|
| Linear Regression | `lr.pkl` | Fits a straight line through data |
| Ridge | `ridge.pkl` | Linear regression with L2 regularization (prevents overfitting) |
| Decision Tree | `dt.pkl` | Tree of yes/no questions to reach prediction |
| Random Forest | `rf.pkl` | 100s of Decision Trees averaged together |
| Gradient Boosting | `gb.pkl` | Trees built sequentially, each fixing previous errors |
| KNN | `knn.pkl` | Finds K most similar patients and averages their charges |

### Feature order (IMPORTANT):
```python
# Must match exactly what StandardScaler was fitted on
[age, sex, bmi, children, smoker, region]
```

### How to retrain models:
```bash
python ml.py
```
This regenerates all `.pkl` files in the project folder.

---

## 🌐 Flask Backend (`app.py`)

### Routes:

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Renders the main form (index.html) |
| `/predict` | POST | Takes form input, scales it, runs all models, returns prediction |

### How prediction works:
```python
# 1. Collect form inputs
age, sex, bmi, children, smoker, region = request.form values

# 2. Build feature array (correct order!)
data = np.array([[age, sex, bmi, children, smoker, region]])

# 3. Scale using saved scaler
data = scaler.transform(data)

# 4. Predict using selected model (or average all)
result = model.predict(data)[0]
```

### AJAX vs Form POST:
- If request has header `X-Requested-With: XMLHttpRequest` → returns **JSON** (used by live UI)
- Otherwise → returns rendered **HTML page** (fallback)

### JSON response format:
```json
{
  "prediction": 12540.25,
  "model_name": "Random Forest",
  "all_predictions": {
    "lr": 11200.50,
    "ridge": 11180.30,
    "dt": 13000.00,
    "rf": 12540.25,
    "gb": 12300.10,
    "knn": 11900.75
  },
  "model_labels": {
    "lr": "Linear Regression",
    ...
  }
}
```

---

## 🎨 Frontend UI (`templates/index.html`)

### Features:
- **Two-panel layout** — form on left, results on right
- **BMI Auto-Calculator** — enter weight (kg) + height (cm) → BMI calculated automatically with color-coded category bar
- **Live AJAX Prediction** — no page reload, uses JavaScript `fetch()` to POST to `/predict`
- **Model Comparison Chart** — Chart.js bar chart showing all 6 model predictions side by side
- **Min / Avg / Max stats** — shown below the chart

### Algorithm options in UI:
- ⚡ Average (all 6 models averaged) ← default
- Linear Regression
- Ridge
- Decision Tree
- Random Forest
- Gradient Boosting
- KNN

---

## ⚙️ How to Run Locally

### Step 1 — Clone the repo
```bash
git clone https://github.com/Kshitij-Mangal/medical-cost-predictor.git
cd medical-cost-predictor
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train models (if .pkl files are missing)
```bash
python ml.py
```

### Step 4 — Run Flask app
```bash
python app.py
```

### Step 5 — Open in browser
```
http://localhost:5000
```

---

## 🚀 Deployment (Render.com)

### Files used for deployment:

| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages to install |
| `Procfile` | Tells Render how to start the app (`gunicorn app:app`) |
| `render.yaml` | Render service configuration |
| `runtime.txt` | Pins Python version to 3.10.5 |

### Deployed at:
```
https://medical-cost-predictor-b95i.onrender.com
```

### ⚠️ Free tier note:
Render's free tier **spins down after 15 min of inactivity**. First request after sleep takes ~30-60 seconds to wake up. Use [cron-job.org](https://cron-job.org) to ping the URL every 10 minutes to keep it awake.

---

## 📦 Dependencies

```
flask>=3.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
gunicorn>=21.2.0
```

Frontend uses CDN (no install needed):
- [Chart.js 4.4.1](https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js)
- [Google Fonts — DM Serif Display + DM Sans](https://fonts.google.com)

---

## 🐛 Issues Fixed During Development

| Issue | Fix |
|-------|-----|
| `sklearn not installed` | Added to requirements.txt |
| `Flask not running` | Added `app.run()` at bottom of app.py |
| `404 error` | Fixed route order in Flask |
| `Feature mismatch (5 vs 6 inputs)` | Added missing `region` field |
| `Wrong feature order` | Fixed to `[age, sex, bmi, children, smoker, region]` |
| `Scaling missing in Flask` | Added `scaler.transform()` before prediction |
| `scaler.pkl not found` | Renamed `scaler.pickle` → `scaler.pkl` |
| `Python 3.14 on Render` | Added `runtime.txt` to pin Python 3.10.5 |
| `render.yaml red line` | Removed invalid `envVars` block, added `plan: free` |

---

## 💡 Skills Used

| Category | Technologies |
|----------|-------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Saving | Pickle |
| Web Backend | Flask, Gunicorn |
| Frontend | HTML, CSS, JavaScript |
| Data Visualization | Chart.js |
| Version Control | Git, GitHub |
| Deployment | Render.com |

---

## 🔮 Future Improvements

- [ ] Add SHAP values for model explainability
- [ ] Add prediction history with database (SQLite/PostgreSQL)
- [ ] Add Deep Learning model (Neural Network)
- [ ] Dockerize the application
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Add unit tests for ML pipeline

---

## 👤 Author

**Kshitij Mangal**  
GitHub: [@Kshitij-Mangal](https://github.com/Kshitij-Mangal)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).