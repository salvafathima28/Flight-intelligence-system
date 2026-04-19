# ✈️ Flight Intelligence Prediction System

A multi-stage machine learning system that predicts **flight delays, delay duration, delay causes, and cancellation risk** using real-world airline data.

---

## 🚀 Project Overview

This project builds an **end-to-end Flight Intelligence System** using U.S. airline performance data.
Instead of a single prediction, it uses a **4-stage machine learning pipeline** to generate meaningful, actionable insights.

### 🎯 Final Output

* Delay Probability
* Expected Delay (minutes)
* Likely Delay Cause
* Cancellation Risk

---

## 🧠 System Architecture

```
Input Features
      ↓
Stage 1 → Delay Classification
      ↓
If Delayed →
   ├── Stage 2 → Delay Regression
   └── Stage 3 → Cause Classification
      ↓
Parallel:
Stage 4 → Cancellation Prediction
      ↓
Final Output
```

---

## 📊 Dataset

* Source: Bureau of Transportation Statistics (BTS)
* Original Data: 12 monthly CSV files
* ✅ Merged Dataset: `full_year.csv`
* Records: ~479,000+ flights

### ⚠️ Important Note

Due to large file size:

* Only compressed dataset is uploaded: `full_year.zip`
* You must extract it before running the project

---

## 📂 Project Structure

```
├── Flight.ipynb              # Model training notebook
├── app.py                   # Streamlit application
├── README.md
├── full_year.zip            # Compressed dataset
├── full_year.csv            # (After extraction)
├── models/                  # Generated after training
│   ├── stage1_delay_classifier.pkl
│   ├── stage2_delay_regressor.pkl
│   ├── stage3_cause_classifier.pkl
│   ├── stage4_cancellation_classifier.pkl
│   ├── encoders.pkl
│   └── metrics.json
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Extract Dataset

Unzip the dataset:

```
full_year.zip → full_year.csv
```

Place `full_year.csv` in the root project folder.

---

### 4. ⚠️ Update Dataset Path (IMPORTANT)

In both files:

* `app.py`
* `Flight.ipynb`

Change:

```python
"alldata/full_year.csv"
```

To:

```python
"full_year.csv"
```

---

### 5. Train Models (MANDATORY STEP)

Run the notebook:

```
Flight.ipynb
```

This will:

* Clean and preprocess the dataset
* Train all 4 machine learning models
* Save models into the `models/` folder

⚠️ If this step is skipped, the app will not run.

---

### 6. Run the Application

```bash
streamlit run app.py
```

---

## 🔮 Features

* 📊 Interactive dashboard with insights
* 🔮 Real-time flight prediction system
* 📈 Model performance visualization
* 🧠 Multi-stage ML pipeline
* 💡 Smart recommendations based on predictions

---

## 🤖 Models Used

| Stage   | Task                    | Model             |
| ------- | ----------------------- | ----------------- |
| Stage 1 | Delay Classification    | Random Forest     |
| Stage 2 | Delay Regression        | Random Forest     |
| Stage 3 | Cause Classification    | Gradient Boosting |
| Stage 4 | Cancellation Prediction | Random Forest     |

---

## 📈 Model Performance

* Delay Classification: ~90% Accuracy
* Delay Regression: MAE ≈ 11 minutes, R² ≈ 0.94
* Cause Classification: ~66% Accuracy
* Cancellation Prediction: ~96% Accuracy

---

## ⚠️ Important Notes

* Models are not pre-trained in this repository
* You must run `Flight.ipynb` once to generate models
* Dataset path must be updated before running
* Large dataset is provided in compressed format

---

## 💡 Future Improvements

* Integrate XGBoost for improved performance
* Deploy the application (Streamlit Cloud / Render)
* Add real-time flight API integration
* Enhance feature engineering

---

## 👩‍💻 Author

**Salva Fathima**

---
