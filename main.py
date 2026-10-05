import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATA
# ============================================================

# Read CSV file using Pandas
df = pd.read_csv("students.csv")


# Display complete dataset
print("\n========== DATASET ==========")
print(df)


# Display number of rows and columns
print("\n========== DATASET SHAPE ==========")
print(df.shape)


# Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns)


# Display information about columns, data types and non-null values
print("\n========== DATA INFORMATION ==========")
df.info()


# Display statistical summary of numerical columns
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# Check missing values in each column
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# Check duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# Display unique departments
print("\n========== UNIQUE DEPARTMENTS ==========")
print(df["Department"].unique())


# Count students in each department
print("\n========== STUDENTS PER DEPARTMENT ==========")
print(df["Department"].value_counts())


# ============================================================
# 2. CREATE NEW COLUMNS
# ============================================================

# Calculate average marks of Math, Python and English
df["Average_Marks"] = (
    df["Math"] + df["Python"] + df["English"]
) / 3


# Calculate total marks out of 300
df["Total_Marks"] = (
    df["Math"] + df["Python"] + df["English"]
)


# Classify attendance as Low or Good
df["Attendance_Status"] = df["Attendance"].apply(
    lambda x: "Low" if x < 75 else "Good"
)


# Classify students according to their average marks
df["Performance"] = df["Average_Marks"].apply(
    lambda x:
    "Excellent" if x >= 90
    else "Good" if x >= 80
    else "Average" if x >= 70
    else "Poor"
)


# Display dataset after adding new columns
print("\n========== DATA WITH NEW COLUMNS ==========")
print(df)


# ============================================================
# 3. NUMPY ANALYSIS
# ============================================================

# Convert subject marks from Pandas DataFrame
# into a NumPy array
marks = df[
    ["Math", "Python", "English"]
].to_numpy()


print("\n========== NUMPY MARKS ARRAY ==========")
print(marks)


# Display shape of NumPy array
print("\n========== NUMPY ARRAY SHAPE ==========")
print(marks.shape)


# Calculate overall average of all subject marks
print("\n========== OVERALL AVERAGE ==========")
print(np.mean(marks))


# Calculate subject-wise average
# axis=0 means column-wise calculation
subject_avg = np.mean(marks, axis=0)

print("\n========== SUBJECT-WISE AVERAGE ==========")
print(subject_avg)


# Calculate student-wise average
# axis=1 means row-wise calculation
student_avg = np.mean(marks, axis=1)

print("\n========== STUDENT-WISE AVERAGE ==========")
print(student_avg)


# Add NumPy calculated average to DataFrame
df["NumPy_Average"] = student_avg

print("\n========== STUDENT + NUMPY AVERAGE ==========")
print(df[["Name", "NumPy_Average"]])


# Find highest marks in each subject
print("\n========== HIGHEST MARKS BY SUBJECT ==========")
print(np.max(marks, axis=0))


# Find lowest marks in each subject
print("\n========== LOWEST MARKS BY SUBJECT ==========")
print(np.min(marks, axis=0))


# Calculate standard deviation of each subject
print("\n========== STANDARD DEVIATION BY SUBJECT ==========")
print(np.std(marks, axis=0))


# Display subject averages clearly
print("\n========== SUBJECT AVERAGES ==========")
print("Math Average   :", subject_avg[0])
print("Python Average :", subject_avg[1])
print("English Average:", subject_avg[2])


# ============================================================
# 4. ACADEMIC RISK ANALYSIS
# ============================================================

# Students are considered at academic risk if:
# Average marks are below 70 OR attendance is below 75
risk_students = df.query(
    "Average_Marks < 70 or Attendance < 75"
)


print("\n========== ACADEMIC RISK STUDENTS ==========")

print(
    risk_students[
        [
            "Name",
            "Department",
            "Attendance",
            "Average_Marks",
            "Performance"
        ]
    ]
)


# ============================================================
# 5. DEPARTMENT ANALYSIS
# ============================================================

# Calculate department-wise average:
# Average Marks, Attendance and Study Hours
department_analysis = (
    df.groupby("Department")[
        [
            "Average_Marks",
            "Attendance",
            "Study_Hours"
        ]
    ].mean()
)


print("\n========== DEPARTMENT ANALYSIS ==========")
print(department_analysis)


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

# Calculate correlation between Study Hours
# and Average Marks
study_marks_correlation = df[
    ["Study_Hours", "Average_Marks"]
].corr()


print("\n========== STUDY HOURS vs AVERAGE MARKS ==========")
print(study_marks_correlation)


# ============================================================
# 7. TOP-PERFORMING STUDENT
# ============================================================

# Find index of student with highest average marks
top_student_index = df["Average_Marks"].idxmax()


print("\n========== TOP-PERFORMING STUDENT ==========")
print(df.loc[top_student_index])


# ============================================================
# 8. ACADEMIC RISK PERCENTAGE
# ============================================================

# Count students at academic risk
risk_count = len(risk_students)


# Count total students
total_students = len(df)


# Calculate percentage of students at risk
risk_percentage = (
    risk_count / total_students
) * 100


print("\n========== ACADEMIC RISK SUMMARY ==========")
print("Risk Students   :", risk_count)
print("Total Students  :", total_students)
print("Risk Percentage :", risk_percentage)


# ============================================================
# 9. WEAKEST SUBJECT
# ============================================================

# Find index of subject having the lowest average
weakest_subject_index = np.argmin(subject_avg)


# Create NumPy array containing subject names
subjects = np.array([
    "Math",
    "Python",
    "English"
])


# Find weakest subject using its index
weakest_subject = subjects[weakest_subject_index]


print("\n========== WEAKEST SUBJECT ==========")
print("Weakest Subject :", weakest_subject)

print(
    "Average Marks   :",
    subject_avg[weakest_subject_index]
)


# ============================================================
# 10. STUDENT PERFORMANCE RANKING
# ============================================================

# Sort students from highest to lowest average marks
ranking = df.sort_values(
    "Average_Marks",
    ascending=False
)


print("\n========== STUDENT PERFORMANCE RANKING ==========")

print(
    ranking[
        [
            "Name",
            "Department",
            "Average_Marks",
            "Performance"
        ]
    ]
)


# ============================================================
# 11. DATA VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# 11.1 Department-wise Average Marks
# ------------------------------------------------------------

# Create figure
plt.figure(figsize=(8, 5))


# Create bar chart using Seaborn
sns.barplot(
    x=department_analysis.index,
    y=department_analysis["Average_Marks"]
)


# Add title and axis labels
plt.title("Average Marks by Department")
plt.xlabel("Department")
plt.ylabel("Average Marks")


# Add horizontal grid
plt.grid(axis="y", alpha=0.3)


# Save graph as PNG
plt.savefig(
    "visualizations/department_performance.png"
)


# Display graph
plt.show()


# ------------------------------------------------------------
# 11.2 Study Hours vs Average Marks
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))


# Create scatter plot
# Each point represents one student
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Average_Marks"
)


plt.title("Study Hours vs Average Marks")
plt.xlabel("Study Hours")
plt.ylabel("Average Marks")

plt.grid(alpha=0.3)


# Save graph
plt.savefig(
    "visualizations/study_hours_vs_marks.png"
)


plt.show()


# ------------------------------------------------------------
# 11.3 Subject-wise Average Marks
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))


# Create bar chart for subject averages
sns.barplot(
    x=subjects,
    y=subject_avg
)


plt.title("Subject-wise Average Marks")
plt.xlabel("Subject")
plt.ylabel("Average Marks")

plt.grid(axis="y", alpha=0.3)


# Save graph
plt.savefig(
    "visualizations/subject_performance.png"
)


plt.show()


# ------------------------------------------------------------
# 11.4 Attendance vs Average Marks
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))


# Create scatter plot
sns.scatterplot(
    data=df,
    x="Attendance",
    y="Average_Marks"
)


plt.title("Attendance vs Average Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Average Marks")

plt.grid(alpha=0.3)


# Save graph
plt.savefig(
    "visualizations/attendance_vs_marks.png"
)


plt.show()


# ------------------------------------------------------------
# 11.5 Performance Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))


# Count number of students in each performance category
sns.countplot(
    data=df,
    x="Performance"
)


plt.title("Student Performance Distribution")
plt.xlabel("Performance Level")
plt.ylabel("Number of Students")

plt.grid(axis="y", alpha=0.3)


# Save graph
plt.savefig(
    "visualizations/performance_distribution.png"
)


plt.show()


# ------------------------------------------------------------
# 11.6 Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))


# Select important numerical columns
# and calculate their correlations
correlation = df[
    [
        "Attendance",
        "Study_Hours",
        "Assignments",
        "Math",
        "Python",
        "English",
        "Average_Marks",
        "Previous_GPA"
    ]
].corr()


# Create correlation heatmap
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)


plt.title("Correlation Heatmap")


# Save heatmap
plt.savefig(
    "visualizations/correlation_heatmap.png"
)


plt.show()