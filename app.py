from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────
lr    = pickle.load(open("lr.pkl",    "rb"))
dt    = pickle.load(open("dt.pkl",    "rb"))
rf    = pickle.load(open("rf.pkl",    "rb"))
gb    = pickle.load(open("gb.pkl",    "rb"))
ridge = pickle.load(open("ridge.pkl", "rb"))
knn   = pickle.load(open("knn.pkl",   "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))   # ← was missing before

MODELS = {
    "lr":    ("Linear Regression", lr),
    "ridge": ("Ridge",             ridge),
    "dt":    ("Decision Tree",     dt),
    "rf":    ("Random Forest",     rf),
    "gb":    ("Gradient Boosting", gb),
    "knn":   ("KNN",               knn),
}


def build_features(form):
    """
    Correct feature order (must match what StandardScaler was fitted on):
    age, sex, bmi, children, smoker, region
    """
    age      = float(form["age"])
    sex      = int(form["sex"])        # male=1, female=0
    bmi      = float(form["bmi"])
    children = int(form["children"])
    smoker   = int(form["smoker"])     # yes=1, no=0
    region   = int(form["region"])     # southwest=1, southeast=0, others=3
    return np.array([[age, sex, bmi, children, smoker, region]])


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles both:
      - AJAX requests (returns JSON)  ← new live-prediction path
      - Regular form POST (returns rendered page)  ← fallback
    """
    try:
        raw   = build_features(request.form)
        data  = scaler.transform(raw)           # ← scale before predict
        algo  = request.form.get("algo", "avg")

        # All-models predictions (used for chart and avg)
        all_preds = {
            key: round(float(model.predict(data)[0]), 2)
            for key, (_, model) in MODELS.items()
        }

        if algo == "avg":
            result     = round(sum(all_preds.values()) / len(all_preds), 2)
            model_name = "Average of All Models"
        else:
            result     = all_preds[algo]
            model_name = MODELS[algo][0]

        # AJAX path
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({
                "prediction": result,
                "model_name": model_name,
                "all_predictions": all_preds,
                "model_labels": {k: v[0] for k, v in MODELS.items()},
            })

        # Regular form POST fallback
        return render_template(
            "index.html",
            prediction=f"${result:,.2f}",
            model_name=model_name,
            all_predictions=all_preds,
            model_labels={k: v[0] for k, v in MODELS.items()},
        )

    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(e)}), 400
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)