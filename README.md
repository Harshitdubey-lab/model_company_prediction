# 👨‍💼 Employee Attrition Prediction

An end-to-end **Machine Learning project for Employee Attrition Prediction** using Python, Pandas, Scikit-learn, XGBoost, Seaborn, Matplotlib, and Streamlit.

The project analyzes employee-related factors and builds classification models to estimate whether an employee belongs to the **Attrition** or **No Attrition** class.

> **Note:** The dataset used in this project shows very weak predictive signal. Therefore, the model results should be interpreted as a machine-learning experiment and analytical demonstration rather than a reliable real-world employee-risk prediction system.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [Dataset](#-dataset)
* [Dataset Features](#-dataset-features)
* [Project Workflow](#-project-workflow)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Data Preprocessing](#-data-preprocessing)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Model Comparison](#-model-comparison)
* [Feature Importance](#-feature-importance)
* [Confusion Matrix](#-confusion-matrix)
* [ROC Curve](#-roc-curve)
* [Cross Validation](#-cross-validation)
* [Streamlit Application](#-streamlit-application)
* [Project Structure](#-project-structure)
* [Technologies Used](#-technologies-used)
* [Installation](#-installation)
* [How to Run](#-how-to-run)
* [Example Prediction](#-example-prediction)
* [Key Findings](#-key-findings)
* [Limitations](#-limitations)
* [Future Improvements](#-future-improvements)
* [Business Use Case](#-business-use-case)
* [Conclusion](#-conclusion)
* [Author](#-author)
* [License](#-license)

---

# 📖 Project Overview

Employee attrition is an important challenge for organizations because employee turnover can increase recruitment costs, training costs, productivity loss, and workload for existing employees.

This project uses employee data to investigate whether machine-learning techniques can identify patterns associated with employee attrition.

The complete project covers:

* Data loading
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Feature analysis
* Categorical encoding
* Train-test splitting
* Feature scaling
* Machine learning model training
* Model evaluation
* Model comparison
* Feature importance analysis
* Cross-validation
* Model saving
* Streamlit deployment

---

# ❓ Problem Statement

The objective of this project is to build a classification system that predicts whether an employee belongs to:

* **No Attrition**
* **Attrition**

The project evaluates multiple machine-learning algorithms and compares their performance using metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Because employee attrition is a binary classification problem, the target variable is converted into:

```text
No  → 0
Yes → 1
```

---

# 🎯 Objectives

The main objectives of this project are:

1. Understand the employee dataset.
2. Identify data-quality issues.
3. Perform exploratory data analysis.
4. Analyze relationships between employee attributes and attrition.
5. Prepare categorical and numerical features for machine learning.
6. Train multiple classification models.
7. Compare model performance.
8. Analyze feature importance.
9. Validate the model using cross-validation.
10. Save the trained model.
11. Build an interactive Streamlit application.
12. Provide an easy-to-use employee attrition prediction interface.

---

# 📊 Dataset

The project uses an employee attrition dataset containing:

* **10,000 employee records**
* **26 original columns**
* Employee demographic information
* Job-related information
* Salary information
* Work-related information
* Satisfaction-related information
* Attrition status

### Dataset Quality

| Check                  | Result |
| ---------------------- | -----: |
| Total Records          | 10,000 |
| Original Features      |     26 |
| Missing Values         |      0 |
| Duplicate Rows         |      0 |
| Duplicate Employee IDs |      0 |
| Attrition = No         |  8,003 |
| Attrition = Yes        |  1,997 |

### Attrition Distribution

```text
No  : 8003 (80.03%)
Yes : 1997 (19.97%)
```

This represents an approximately **80:20 class distribution**.

---

# 🧾 Dataset Features

The dataset contains the following features:

| Feature                       | Description                     |
| ----------------------------- | ------------------------------- |
| Employee_ID                   | Unique employee identifier      |
| Age                           | Employee age                    |
| Gender                        | Employee gender                 |
| Marital_Status                | Marital status                  |
| Department                    | Employee department             |
| Job_Role                      | Employee job role               |
| Job_Level                     | Employee job level              |
| Monthly_Income                | Monthly income                  |
| Hourly_Rate                   | Hourly rate                     |
| Years_at_Company              | Years spent at company          |
| Years_in_Current_Role         | Years in current role           |
| Years_Since_Last_Promotion    | Years since last promotion      |
| Work_Life_Balance             | Work-life balance rating        |
| Job_Satisfaction              | Job satisfaction rating         |
| Performance_Rating            | Performance rating              |
| Training_Hours_Last_Year      | Training hours                  |
| Overtime                      | Whether employee works overtime |
| Project_Count                 | Number of projects              |
| Average_Hours_Worked_Per_Week | Average weekly working hours    |
| Absenteeism                   | Absenteeism measure             |
| Work_Environment_Satisfaction | Workplace satisfaction          |
| Relationship_with_Manager     | Manager relationship rating     |
| Job_Involvement               | Job involvement rating          |
| Distance_From_Home            | Distance from home              |
| Number_of_Companies_Worked    | Number of previous companies    |
| Attrition                     | Target variable                 |

---

# 🔄 Project Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Analysis
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Cross Validation
   ↓
Model Saving
   ↓
Streamlit Application
```

---

# 🔍 Exploratory Data Analysis

Several EDA techniques were performed to understand the dataset.

## 1. Dataset Shape

```python
print(df.shape)
```

Output:

```text
(10000, 26)
```

---

## 2. Missing Value Analysis

```python
print(df.isnull().sum())
```

Result:

```text
No missing values found.
```

---

## 3. Duplicate Analysis

```python
print(df.duplicated().sum())
```

Result:

```text
0
```

No duplicate records were found.

---

## 4. Attrition Distribution

```python
print(df['Attrition'].value_counts())
```

Output:

```text
No     8003
Yes    1997
```

Percentage distribution:

```text
No     80.03%
Yes    19.97%
```

The dataset is therefore imbalanced toward the **No Attrition** class.

---

# 📈 Data Visualization

The project includes several visualizations.

### Attrition Distribution

A count plot was used to understand the distribution of employees across the two target classes.

### Categorical Analysis

The following variables were analyzed against attrition:

* Overtime
* Job Role
* Department
* Gender
* Marital Status

Example:

```python
sns.countplot(x='Overtime', hue='Attrition', data=df)
plt.title("Overtime vs Attrition")
plt.show()
```

### Numerical Analysis

The following variables were analyzed using boxplots:

* Job Satisfaction
* Work Life Balance
* Monthly Income
* Years at Company
* Years in Current Role
* Absenteeism
* Distance From Home

---

# 🔥 Correlation Analysis

A correlation heatmap was created to examine relationships among numerical variables.

```python
plt.figure(figsize=(12, 8))

sns.heatmap(
    df[numerical_columns].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()
```

Correlation analysis helps identify relationships between numerical variables and possible feature redundancy.

---

# 🧹 Data Preprocessing

## 1. Remove Employee ID

`Employee_ID` is an identifier rather than a meaningful predictive feature.

Therefore:

```python
df = df.drop(columns=['Employee_ID'])
```

---

## 2. Separate Features and Target

```python
X = df.drop(columns=['Attrition'])
y = df['Attrition']
```

---

## 3. Encode Target Variable

```python
y = y.map({
    'No': 0,
    'Yes': 1
})
```

Result:

```text
0 → No Attrition
1 → Attrition
```

---

## 4. One-Hot Encoding

Categorical variables were converted into numerical features using one-hot encoding.

```python
categorical_features = [
    'Gender',
    'Marital_Status',
    'Department',
    'Job_Role',
    'Overtime'
]

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True
)
```

---

## 5. Train-Test Split

A stratified 80/20 train-test split was used.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This resulted in:

```text
Training data → 80%
Testing data  → 20%
```

---

# 🤖 Machine Learning Models

Three machine-learning algorithms were evaluated.

## 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

```python
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42
)
```

Feature scaling was applied using `StandardScaler`.

---

## 2. Random Forest

Random Forest was used as a tree-based ensemble classifier.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

---

## 3. XGBoost

XGBoost was used as a gradient-boosting classification model.

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
```

---

# 📊 Model Evaluation

The models were evaluated using:

### Accuracy

Percentage of correctly classified observations.

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

For an imbalanced dataset, **accuracy alone should not be used to judge model quality**.

---

# 🏆 Model Comparison

| Model               |   Accuracy | Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | --------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.5135 |    0.2004 | **0.4812** | **0.2830** | **0.5054** |
| Random Forest       | **0.8005** |    0.0000 |     0.0000 |     0.0000 |     0.5011 |
| XGBoost             |     0.6325 |    0.2083 |     0.3008 |     0.2462 |     0.4983 |

---

# ⚠️ Important Model Observation

Although Random Forest achieved approximately **80% accuracy**, it predicted every test observation as **No Attrition**.

Prediction distribution:

```text
Predicted No Attrition → 2000
Predicted Attrition    → 0
```

Therefore:

```text
Accuracy = 80.05%
Recall   = 0%
F1 Score = 0%
```

The 80% accuracy is primarily explained by the 80:20 class distribution in the dataset.

This demonstrates why evaluating an imbalanced classification problem using accuracy alone can be misleading.

---

# 🔬 Mutual Information Analysis

Mutual Information was also used to examine relationships between features and the target.

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(
    X_encoded,
    y,
    random_state=42
)
```

Top features included:

| Feature                       | MI Score |
| ----------------------------- | -------: |
| Department_IT                 | 0.007137 |
| Performance_Rating            | 0.005774 |
| Years_in_Current_Role         | 0.005548 |
| Job_Involvement               | 0.004883 |
| Overtime_Yes                  | 0.004864 |
| Gender_Male                   | 0.004211 |
| Relationship_with_Manager     | 0.003135 |
| Number_of_Companies_Worked    | 0.002830 |
| Average_Hours_Worked_Per_Week | 0.002815 |
| Age                           | 0.002776 |

The scores are very low overall, indicating that the available features provide limited measurable information about the target in this dataset.

---

# 🌳 XGBoost Feature Importance

The XGBoost model was also used to inspect feature importance.

Top features included:

| Feature                    | Importance |
| -------------------------- | ---------: |
| Job_Role_Manager           |   0.037703 |
| Job_Role_Assistant         |   0.036668 |
| Job_Level                  |   0.035731 |
| Number_of_Companies_Worked |   0.035385 |
| Department_IT              |   0.035230 |
| Overtime_Yes               |   0.034941 |
| Marital_Status_Single      |   0.034677 |
| Relationship_with_Manager  |   0.033876 |
| Job_Role_Executive         |   0.033836 |
| Gender_Male                |   0.033700 |
| Performance_Rating         |   0.033570 |
| Project_Count              |   0.033560 |
| Department_Sales           |   0.033217 |
| Department_HR              |   0.032916 |
| Work_Life_Balance          |   0.032912 |

The importance values are relatively close to each other. These values should therefore be interpreted as model-specific feature importance rather than evidence of causal relationships.

---

# 📉 Confusion Matrix

The XGBoost confusion matrix was:

```text
True Negative  = 1145
False Positive = 456
False Negative = 279
True Positive  = 120
```

Therefore:

```text
Recall = 120 / (120 + 279)
       ≈ 30.08%
```

This means the XGBoost model correctly identified approximately 30% of the actual Attrition cases in the test set.

---

# 📈 ROC Curve

The ROC curve was generated for XGBoost.

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob_xgb
)

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc_score(y_test, y_prob_xgb):.3f})"
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend()

plt.show()
```

The resulting ROC-AUC was approximately:

```text
0.498
```

An AUC close to 0.50 indicates discrimination close to random performance on this dataset.

---

# 🔁 Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    xgb_model,
    X_encoded,
    y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
```

Results:

```text
Fold 1 → 0.4963
Fold 2 → 0.5057
Fold 3 → 0.5007
Fold 4 → 0.5010
Fold 5 → 0.5119
```

Mean ROC-AUC:

```text
0.5031
```

Standard deviation:

```text
0.0053
```

The cross-validation scores remain close to 0.50 across folds, indicating that the predictive signal is weak and consistent with the test-set results.

---

# 💻 Streamlit Application

An interactive **Streamlit web application** was created for employee attrition prediction.

The application allows users to enter employee information such as:

* Age
* Gender
* Marital Status
* Department
* Job Role
* Job Level
* Monthly Income
* Hourly Rate
* Years at Company
* Years in Current Role
* Years Since Last Promotion
* Work-Life Balance
* Job Satisfaction
* Performance Rating
* Training Hours
* Overtime
* Project Count
* Average Working Hours
* Absenteeism
* Work Environment Satisfaction
* Relationship with Manager
* Job Involvement
* Distance From Home
* Number of Companies Worked

The application then:

```text
User Input
    ↓
DataFrame Creation
    ↓
One-Hot Encoding
    ↓
Feature Alignment
    ↓
Feature Scaling
    ↓
Model Prediction
    ↓
Attrition Result
    ↓
Estimated Probability
```

---

# 🗂️ Project Structure

A recommended repository structure is:

```text
Employee-Attrition-Prediction/
│
├── data/
│   └── employee_attrition_dataset_10000.csv
│
├── notebooks/
│   └── employee_attrition_analysis.ipynb
│
├── models/
│   ├── logistic_attrition_model.pkl
│   ├── attrition_scaler.pkl
│   └── attrition_features.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

> If the dataset or model files are large, consider using Git LFS or another appropriate artifact-storage approach rather than committing very large files directly to Git. GitHub recommends Git LFS for large files.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Environment

* Google Colab
* Jupyter Notebook
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Employee-Attrition-Prediction.git
```

Move into the project directory:

```bash
cd Employee-Attrition-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
streamlit
joblib
```

---

# ▶️ How to Run

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔮 Example Prediction

The Streamlit application provides a prediction similar to:

```text
Prediction Result

Attrition Prediction: NO

Estimated Attrition Probability: XX.XX%
```

or:

```text
Prediction Result

Attrition Prediction: YES

Estimated Attrition Probability: XX.XX%
```

The probability shown by the application comes from the trained classification model.

Because the evaluated model has weak ROC-AUC performance, this probability should **not** be interpreted as a reliable real-world probability of an employee leaving.

---

# 🔎 Key Findings

The analysis produced several important findings:

### 1. Dataset is imbalanced

Approximately:

```text
80.03% → No Attrition
19.97% → Attrition
```

### 2. Accuracy is not sufficient

Random Forest achieved around 80% accuracy but identified **zero Attrition cases**.

Therefore, accuracy alone can give a misleading impression of model performance.

### 3. Logistic Regression

Logistic Regression produced the highest recall among the three tested models:

```text
Recall ≈ 48.12%
```

However, its ROC-AUC was only approximately:

```text
0.505
```

### 4. XGBoost

XGBoost produced:

```text
Accuracy ≈ 63.25%
Precision ≈ 20.83%
Recall ≈ 30.08%
F1 ≈ 24.62%
ROC-AUC ≈ 0.498
```

### 5. Cross-validation

The mean XGBoost ROC-AUC across five folds was:

```text
0.5031
```

### 6. Overall observation

Across the tested approaches, the dataset does not demonstrate strong predictive signal for employee attrition.

This is reflected by:

* ROC-AUC values around 0.50
* Very low mutual-information scores
* Similar feature distributions between target classes
* Weak cross-validation performance

---

# ⚠️ Limitations

This project has several limitations.

## 1. Weak Predictive Signal

The evaluated models achieved ROC-AUC values close to 0.50.

Therefore, the current dataset does not support strong predictive performance.

## 2. Dataset Characteristics

The dataset may not capture all real-world factors affecting employee turnover, such as:

* Career opportunities
* Compensation satisfaction
* Leadership changes
* Organizational culture
* Personal circumstances
* External job market conditions
* Employee engagement
* Burnout
* Workplace relationships

## 3. Class Imbalance

The target variable contains significantly more `No Attrition` observations than `Yes Attrition`.

Therefore, accuracy can be misleading.

## 4. Model Generalization

The model should not be assumed to generalize to other organizations or employee populations without validation on representative external data.

## 5. Prediction Interpretation

The model output should be considered analytical support rather than a definitive prediction of an employee's future decision.

---

# 🚀 Future Improvements

Future versions of the project could include:

* Collecting a larger real-world dataset
* Adding more meaningful employee attributes
* Hyperparameter tuning
* Feature engineering
* Threshold optimization
* Calibration of predicted probabilities
* SHAP explainability
* Precision-Recall curve analysis
* Cost-sensitive learning
* Advanced ensemble models
* Model monitoring
* External validation
* Improved Streamlit UI
* Database integration
* Cloud deployment
* Authentication and role-based access
* Automated model retraining

---

# 💼 Business Use Case

In a real-world scenario, an employee analytics system could help HR teams:

* Monitor workforce trends
* Understand employee engagement patterns
* Identify areas requiring organizational attention
* Analyze factors associated with employee turnover
* Support workforce planning

However, employee-related predictions should be used carefully and should not automatically determine employment decisions.

The purpose of such a system should be to provide **aggregate analytical insights and decision support**, rather than make automated decisions about individual employees.

---

# 🔐 Responsible AI Considerations

Employee attrition prediction involves sensitive organizational decisions.

Important considerations include:

* Protecting employee privacy
* Avoiding discriminatory use
* Monitoring model bias
* Validating predictions on representative data
* Avoiding automated employment decisions
* Keeping human oversight
* Using predictions as supporting information rather than definitive judgments

---

# 📚 Learning Outcomes

Through this project, the following concepts were implemented:

* Python for data analysis
* Pandas DataFrame operations
* NumPy
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Categorical encoding
* Feature scaling
* Train-test splitting
* Stratified sampling
* Class imbalance handling
* Logistic Regression
* Random Forest
* XGBoost
* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC
* ROC Curve
* Mutual Information
* Feature Importance
* Cross Validation
* Model Persistence
* Streamlit deployment

---

# 🧪 Model Development Summary

```text
Dataset
   │
   ├── 10,000 Employees
   │
   ├── Data Cleaning
   │
   ├── EDA
   │
   ├── Feature Engineering
   │
   ├── One-Hot Encoding
   │
   ├── Train/Test Split
   │
   ├── Logistic Regression
   │
   ├── Random Forest
   │
   ├── XGBoost
   │
   ├── Model Evaluation
   │
   ├── Cross Validation
   │
   ├── Model Saving
   │
   └── Streamlit Application
```

---

# 📌 Conclusion

This project demonstrates a complete end-to-end machine-learning workflow for employee attrition analysis.

The project covers everything from:

**Data → EDA → Preprocessing → Model Training → Evaluation → Feature Analysis → Cross Validation → Model Saving → Streamlit Application**

The experiments also demonstrate an important machine-learning lesson:

> **A high accuracy score does not necessarily mean that a classification model is useful, especially when the target classes are imbalanced.**

In this dataset, the tested models produced ROC-AUC values close to random performance, showing that additional or better-quality data and stronger features would be required to develop a dependable attrition prediction system.

---

# 👨‍💻 Author

**Harshit Dubey**


---

# 📄 License

This project is available for educational and portfolio purposes.

Add an appropriate open-source license such as the **MIT License** if you want to explicitly permit reuse and modification.

---

# ⭐ If You Find This Project Useful

If you found this project useful for learning machine learning, data analysis, or Streamlit development, consider giving the repository a ⭐ on GitHub.
