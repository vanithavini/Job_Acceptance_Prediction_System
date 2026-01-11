# 🎯 Job Acceptance Prediction System

A complete end-to-end Machine Learning application that predicts whether a candidate will accept a job offer based on academic background, skills, experience, interview performance, and compensation expectations. The system also provides analytical insights through an interactive dashboard.

## 📌 Project Overview

The **Job Acceptance Prediction System** helps HR teams and recruiters:
- Predict job acceptance probability
- Identify high-risk candidates
- Analyze placement trends and KPIs
- Make data-driven hiring decisions

This project follows a **full ML lifecycle**:
- Data preprocessing
- Feature engineering
- Model training
- Database integration
- Model deployment using Streamlit


## 🧠 Problem Statement

Recruiters often face challenges in predicting whether a candidate will accept a job offer. Manual judgment can lead to:
- Offer dropouts
- Increased hiring costs
- Inefficient recruitment cycles

This project uses **Machine Learning classification** to predict job acceptance outcomes and provide actionable insights.


## 📂 Project Structure

job_acceptance_prediction_system/
│
├── job_venv  # create virtual environment
|
├── data/
│ ├── cleaned/
│ │ ├── job_acceptance_cleaned.csv
│ │ └── job_acceptance_f_e_data.csv
│ ├── raw/
|   ├── job_acceptance_backup.csv
|
├── artifacts/
│ ├── model.pkl
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ ├── feature_columns.pkl
│ └── numeric_columns.pkl
│
├── notebooks/
│ ├── eda.ipynb
│ └── feature_engineering.ipynb
│ └── mysql_datastorage.ipynb
│
├── source/
| ├── data_understanding.py
│ └── data_cleaning_preprocessing.py
|
├── ml_main.py
|
├── app/
│ ├── app.py
|
├── requirements.txt
└── README.md


## 📊 Dataset Description

The dataset contains candidate-level information such as:
- Academic performance (SSC, HSC, Degree)
- Skills match percentage
- Certifications
- Experience
- Interview scores
- CTC expectations
- Company tier
- Placement status (Target variable)

## 🧠 Type of Machine Learning Problem

Problem Type: Supervised Learning
Task: Binary Classification

Target Variable: status
1 → Placed
0 → Not Placed

## 🔄 Project Workflow

## 🧩 Data Understanding

Purpose:
- To explore and understand the raw dataset before applying any transformations.
Key Steps:
- Load raw dataset
- Inspect shape, columns, and data types
- Check missing values and duplicates
Validate logical consistency:
- Age vs Experience
- Academic percentage ranges
- Compensation validity
- Review categorical feature consistency
Outcome:
- Identified data quality issues
- Defined cleaning and preprocessing strategy
- Backed up raw dataset for safety


## 🧹 Data Cleaning & Preprocessing

Purpose:
- To make the dataset clean, consistent, and machine-learning ready.
Steps Performed:
- Missing Value Handling
- Numerical: Median imputation
- Categorical: Mode imputation
Logical Corrections:
- Experience cannot be negative or exceed age − 18
- Academic scores clipped between 0–100
- Enforced SSC ≤ HSC ≤ Degree
- No negative CTC, notice period, or employment gap
Categorical Standardization:
- Trimmed spaces
- Uniform text casing
Encoding:
- Label Encoding for categorical features
Feature Scaling:
- StandardScaler applied to numerical features
Output:
- Cleaned dataset saved as job_acceptance_cleaned.csv

## 🔍 Exploratory Data Analysis (EDA)

EDA includes:
- Target variable distribution
- Academic performance vs placement
- Skills match vs job acceptance
- Interview performance analysis
- Company tier impact
- Experience and compensation gap analysis
- Correlation heatmap
- Hypothesis testing:
  - **T-test** → Skills match vs acceptance
  - **Chi-square test** → Company tier vs acceptance


## ⚙️ Feature Engineering

Key engineered features:
- `academic_band`
- `experience_category`
- `skills_match_level`
- `interview_score_avg`
- `interview_performance`
- `placement_probability_score`
- `ctc_gap`
- `ctc_gap_flag`

These features improve model interpretability and performance.


## 🗄️ Database Integration (MySQL)

- Feature-engineered data is stored in **MySQL**
- Enables structured storage and scalability
- Data inserted using batch processing

Database: `job_placement_db`  
Table: `candidate_data`


## 🤖 Machine Learning Model

### 🔹 Problem Type
- **Binary Classification**

### 🔹 Model Used
- **Random Forest Classifier**

### 🔹 Why Random Forest?
- Handles non-linear relationships
- Works well with mixed data types
- Reduces overfitting using ensemble learning
- High accuracy on tabular data

### 🔹 Model Pipeline
1. Label encoding for categorical variables
2. Standard scaling for numerical features
3. Train-test split (80-20)
4. Model training
5. Model evaluation
6. Artifact serialization


## 🌐 Web Application (Streamlit)

### 📊 Dashboard
- Placement rate
- Acceptance rate
- Average interview score
- Skills match analysis
- High-risk candidate identification

### 🤖 Prediction Module
- Real-time candidate input
- Acceptance prediction
- Probability score output


## 🧪 Model Performance

- Accuracy: ~**High accuracy depending on dataset**
- Balanced class handling using `class_weight="balanced"`


## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn |
| Database | MySQL |
| Deployment | Streamlit |
| Serialization | Pickle |


## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
Using GitHub

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Train the Model
python ml_main.py

### 4️⃣ Run Streamlit App
streamlit run app.py


### 📈 Future Enhancements
- Use advanced models (XGBoost, LightGBM)
- Add SHAP explainability
- API deployment using FastAPI
- Authentication for HR users
- Real-time database inference

