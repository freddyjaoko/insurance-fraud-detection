# 🕵️‍♂️ Insurance Fraud Detection

A machine learning-powered system to identify and flag potentially fraudulent insurance claims, complete with data pipelines, model training, evaluation metrics, and an API/streamlit interface for real-world use.

---

## 🚀 Project Overview

Insurance fraud drives significant financial losses. This project tackles the problem by building a robust, end-to-end pipeline to:

* **Ingest** raw claims data,
* **Explore & preprocess** it using EDA techniques and feature engineering,
* **Train** classification models (XGBoost, Random Forest, etc.),
* **Evaluate** performance via precision, recall, F1, and ROC-AUC,
* **Deploy** the best model via API and/or Streamlit dashboard.

Structurally similar to \[Travelers Insurance Fraud repository]\([github.com][1], [github.com][2], [github.com][3], [github.com][4], [github.com][5], [github.com][6]), our project supports both Jupyter-based experimentation and production-ready deployment.

---

## 📁 Repository Structure

```
.
├── data/                 # Raw, processed, and interim datasets
├── notebooks/           # EDA, feature engineering, model training
├── src/
│   ├── data/             # Data loaders & processors
│   ├── features/         # Feature engineering scripts
│   ├── models/           # Model training, tuning, saving
│   ├── api/              # Flask/FastAPI endpoints (predict)
│   └── dashboard/        # Streamlit web interface
├── requirements.txt
├── Dockerfile & docker-compose.yaml
└── README.md
```

---

## 🔧 Key Components

**1. Data Preparation**

* Load raw data (CSV / database)
* Clean, impute, encode categorical variables
* Handle missing values & class imbalance (SMOTE/oversampling)

**2. Exploratory Data Analysis (EDA)**

* Feature–target correlation analysis
* Univariate & bivariate plots, summary statistics

**3. Feature Engineering**

* Create new features
* Drop non-informative columns
* Feature selection via XGBoost importance or statistical tests

**4. Model Training 💡**

* Train/test splits & cross-validation
* Algorithms: XGBoost, RandomForest, Logistic Regression, etc.
* Hyperparameter tuning using GridSearchCV / Optuna

**5. Performance Evaluation**

* Metrics: Precision, Recall, F1-Score, ROC-AUC
* Visuals: Confusion matrices, ROC curves
* Focus on high recall to minimize missed frauds

**6. Model Deployment**

* **API**: Uses Flask or FastAPI to accept JSON inputs and return fraud predictions
* **Streamlit dashboard**: User-friendly interface to upload claim details and view predictions

---

## 🛠️ Installation & Setup

1. **Clone the repo:**

   ```bash
   git clone https://github.com/freddyjaoko/insurance-fraud-detection.git
   cd insurance-fraud-detection
   ```

2. **Install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Prepare data:**

   * Place raw CSVs in `data/raw/`
   * Run preprocessing script:

     ```bash
     python src/data/preprocess.py
     ```

4. **Train and evaluate models:**

   ```bash
   python src/models/train.py
   ```

5. **Run the API server:**

   ```bash
   uvicorn src.api.app:app --reload
   ```

6. **Launch the dashboard:**

   ```bash
   streamlit run src/dashboard/app.py
   ```

---

## 🧪 Usage Examples

### API:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"claim_amount":1000, "policy_holder_age":45, … }'
```

### Dashboard:

* Upload a CSV file of claims or input single claim details via UI to get predictions.

---

## 📊 Metrics & Results

* **Best Model:** XGBoost with > 0.85 AUC, recall > 0.80
* ROC curve and confusion matrix notebooks are available in `notebooks/`
* Prioritizes minimizing false negatives (missed fraud)

---

## 🧩 Technologies & Libraries

* **Data:** Pandas, NumPy
* **Modeling:** scikit-learn, XGBoost, imbalanced-learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** FastAPI, Uvicorn, Streamlit
* **Containerization:** Docker, docker-compose

---

## 📖 References & Inspiration

* Travelers Insurance Fraud Detection (Kaggle/STAT8501)([github.com][6], [github.com][2], [ijettjournal.org][7])
* Insurance-Fraud-Detection End–to–End template([github.com][8])

---

## 🤝 Contributions

Contributions, feedback, and feature suggestions are welcome! To contribute:

1. Fork the repo
2. Feature your branch (`feature/your-feature`)
3. Submit a pull request

Please ensure tests (if available) pass and that your changes align with project structure.

---

## 📄 License

This project is licensed under MIT (see [LICENSE](LICENSE) for details).

---

## ✉️ Contact

For questions or support, open an issue or reach out to my GitHub profile.

---
