import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the raw Dataset
file_path = (r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\data\raw\job_acceptance_backup.csv")
df = pd.read_csv(file_path)

# Quick check
df.head()
df.shape

# Fix Missing Values
# Numerical columns: median imputation
num_cols = df.select_dtypes(include=['int64','float64']).columns
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical columns: mode imputation
cat_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Correct Logical Inconsistencies
# Age vs Experience(Experience cannot be negative)
df['years_of_experience'] = df['years_of_experience'].apply(lambda x: max(x,0))

# Experience cannot exceed age-18 (assuming minimum working age is 18)
df['max_possible_exp'] = df['age_years'] - 18
df['years_of_experience'] = np.where(df['years_of_experience'] > df['max_possible_exp'], df['max_possible_exp'], df['years_of_experience'])
df.drop('max_possible_exp', axis=1, inplace=True)

# Academic Percentages (SSC, HSC, Degree)
for col in ['ssc_percentage','hsc_percentage','degree_percentage']:
    df[col] = df[col].clip(lower=0, upper=100) 

# Ensure SSC <= HSC <= Degree
df['hsc_percentage'] = np.where(df['hsc_percentage'] < df['ssc_percentage'], df['ssc_percentage'], df['hsc_percentage'])
df['degree_percentage'] = np.where(df['degree_percentage'] < df['hsc_percentage'], df['hsc_percentage'], df['degree_percentage'])

# CTC (Compensation)
for col in ['previous_ctc_lpa','expected_ctc_lpa']:
    df[col] = df[col].clip(lower=0) 

df['expected_ctc_lpa'] = np.where(df['expected_ctc_lpa'] < 0.5*df['previous_ctc_lpa'], df['previous_ctc_lpa'], df['expected_ctc_lpa'])

# Notice Period & Employment Gap
df['notice_period_days'] = df['notice_period_days'].apply(lambda x: max(x,0))
df['employment_gap_months'] = df['employment_gap_months'].apply(lambda x: max(x,0))

# Standardize Categorical Labels
categorical_cols = ['gender','degree_specialization','internship_experience','career_switch_willingness',
                    'relevant_experience','company_tier','job_role_match','competition_level',
                    'bond_requirement','layoff_history','relocation_willingness','status']

for col in categorical_cols:
    df[col] = df[col].str.strip().str.title()

# Encode Categorical Features
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 6: Feature Scaling (Numerical)
scaler = StandardScaler()
numerical_cols = ['age_years','ssc_percentage','hsc_percentage','degree_percentage',
                  'technical_score','aptitude_score','communication_score',
                  'skills_match_percentage','certifications_count',
                  'years_of_experience','previous_ctc_lpa','expected_ctc_lpa',
                  'notice_period_days','employment_gap_months']

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Final Check
print("Missing values after cleaning:\n", df.isnull().sum())
print("Dataset shape after cleaning:", df.shape)
print(df.head())

# Saved the cleaned data set
df.to_csv(
    r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\data\cleaned\job_acceptance_cleaned.csv",
    index=False
)

