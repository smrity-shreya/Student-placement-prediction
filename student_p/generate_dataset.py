"""
generate_dataset.py
Synthetic Student Placement Dataset Generator
Author: Project by Smrity Shreya
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

cgpa = np.round(np.random.uniform(5.0, 10.0, N), 2)
internships = np.random.randint(0, 4, N)          # 0–3
projects = np.random.randint(0, 5, N)              # 0–4
tech_score = np.random.randint(50, 101, N)         # 50–100
backlogs = np.random.randint(0, 3, N)              # 0–2

# Realistic placement probability
log_odds = (
    -10.0
    + 1.2  * cgpa
    + 0.8  * internships
    + 0.5  * projects
    + 0.06 * tech_score
    - 1.5  * backlogs
)
prob = 1 / (1 + np.exp(-log_odds))
placed = np.random.binomial(1, prob, N)

df = pd.DataFrame({
    "CGPA": cgpa,
    "Internships": internships,
    "Projects": projects,
    "Technical_Skills_Score": tech_score,
    "Core_Backlogs": backlogs,
    "Placed": placed
})

df.to_csv("placement_data.csv", index=False)
print(f"Dataset saved → placement_data.csv  |  Shape: {df.shape}")
print(df["Placed"].value_counts().rename({1: "Placed", 0: "Not Placed"}))
print(df.head())
