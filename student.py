# student_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------
# Step 1: Load the data
# -------------------------------
math_df = pd.read_csv(r"C:\Users\Arbaz Khan\Downloads\student+performance\student\student-mat.csv", sep=';')
port_df = pd.read_csv(r"C:\Users\Arbaz Khan\Downloads\student+performance\student\student-por.csv", sep=';')

# -------------------------------
# Step 2: Merge the datasets
# -------------------------------
merge_columns = ["school", "sex", "age", "address", "famsize", "Pstatus",
                 "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

merged_df = pd.merge(math_df, port_df, on=merge_columns, suffixes=('_math', '_por'))

print(f"✅ Total merged records: {len(merged_df)}")
print(merged_df[['G3_math', 'G3_por']].describe())

# -------------------------------
# Step 3: Feature Engineering
# -------------------------------
# Add average grade and pass/fail
merged_df['G3_avg'] = (merged_df['G3_math'] + merged_df['G3_por']) / 2
merged_df['pass_math'] = merged_df['G3_math'] >= 10
merged_df['pass_por'] = merged_df['G3_por'] >= 10

# -------------------------------
# Step 4: Visualize Data
# -------------------------------
# Distribution of average grade
sns.histplot(merged_df['G3_avg'], bins=20, kde=True)
plt.title("Distribution of Average Final Grades")
plt.xlabel("Average Grade")
plt.ylabel("Count")
plt.show()

# Study time vs math grade
sns.boxplot(x='studytime_math', y='G3_math', data=merged_df)
plt.title("Study Time vs Math Grade")
plt.show()

# -------------------------------
# Step 5: Build a Simple Model
# -------------------------------
features = ['studytime_math', 'failures_math', 'absences_math']
X = merged_df[features]
y = merged_df['G3_math']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"📈 Linear Regression R² Score: {score:.2f}")
