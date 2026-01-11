import pandas as pd  
import numpy as np   

# Load the dataset
df = pd.read_csv(r"C:\Users\Vanitha\OneDrive\Desktop\HR_Job_Placement_Dataset.csv")

# Quick Check
print("Dataset Shape:", df.shape) 
print("Columns:", df.columns.tolist())  
print("First 5 rows:\n", df.head())  

# Check data types
print("Data types:\n ", df.info())

# Summary statistics for numerical columns
print("Summary Statistics for Numeric columns:\n ", df.describe())

# Check missing values
print("Missing values per column:\n", df.isnull().sum())

# Check duplicates
print("Number of duplicate rows:", df.duplicated().sum())

# Logical consistency checks
# Age Vs experience 
# Check for negative experience
print("Negative experience values:\n", df[df['years_of_experience'] < 0])

# Check for experience > age - 18 
df['max_possible_exp'] = df['age_years'] - 18
invalid_exp = df[df['years_of_experience'] > df['max_possible_exp']]
print("Candidates with impossible experience:\n", invalid_exp[['age_years','years_of_experience']])

# Academic Percentages
# Percentages outside 0-100
invalid_percent = df[(df['ssc_percentage']>100) | (df['ssc_percentage']<0) |
                     (df['hsc_percentage']>100) | (df['hsc_percentage']<0) |
                     (df['degree_percentage']>100) | (df['degree_percentage']<0)]
print("Invalid percentages:\n", invalid_percent)

# SSC <= HSC <= Degree check
academic_issue = df[(df['ssc_percentage'] > df['hsc_percentage']) | (df['hsc_percentage'] > df['degree_percentage'])]
print("Candidates with inconsistent academic progression:\n", academic_issue)

# CTC (Compensation) 
ctc_issue = df[(df['previous_ctc_lpa'] < 0) | (df['expected_ctc_lpa'] < 0)]
print("Negative CTC values:\n", ctc_issue)

large_ctc_drop = df[df['expected_ctc_lpa'] < 0.5*df['previous_ctc_lpa']]
print("Candidates expecting big drop in CTC:\n", large_ctc_drop)

# Notice Period & Employment Gap 
print("Negative notice periods:", df[df['notice_period_days']<0])
print("Negative employment gaps:", df[df['employment_gap_months']<0])

# Categorical Consistency
for col in ['gender','internship_experience','career_switch_willingness']:
    print(col, "unique values:", df[col].unique())

# Save a raw copy of data
df.to_csv(
    r"C:\Users\Vanitha\OneDrive\Documents\GUVI\capstone_project\job_acceptance_prediction_system\data\raw\job_acceptance_backup.csv",
    index=False
)

