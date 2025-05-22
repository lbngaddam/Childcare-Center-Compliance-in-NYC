# Child Care Center Compliance Analysis  Compliance and Violation Prediction

This project analyzes child care center inspection data from the DOHMH (Department of Health and Mental Hygiene) to understand compliance patterns and predict the likelihood of critical violations. The analysis involves data cleaning, exploratory data analysis (EDA), and machine learning modeling.

---

## 📄 Dataset

The primary dataset used is `DOHMH_Childcare_Center_Inspections_20250209.csv`. This dataset contains information about child care center inspections, including center details, permit information, violation rates, inspection dates, and violation details.

---

## 📝 Project Overview

The notebook performs the following key tasks:

1.  **Data Loading and Initial Exploration:**
    * Imports necessary Python libraries (pandas, numpy, matplotlib, seaborn, scikit-learn).
    * Loads the dataset into a pandas DataFrame.
    * Performs initial checks like viewing the head of the data, checking its shape, and getting column information (`.head()`, `.shape`, `.info()`).

2.  **Data Preprocessing and Cleaning:**
    * Identifies columns with a high percentage of missing values (≥ 40%) and drops them (`URL`, `Violation Category`, `Health Code Sub Section`, `Violation Status`).
    * Converts date-related columns to datetime objects.
    * Drops rows where critical information like `Inspection Date`, `Violation Category`, or `Violation Status` is missing.
    * Imputes missing numerical values with the median of their respective columns.
    * Imputes missing categorical values with the mode of their respective columns.
    * The cleaned dataset is saved as `cleaned_child_care_compliance.csv`.

3.  **Exploratory Data Analysis (EDA) 📊:**
    * **Inspections Over Time:** Visualizes the number of inspections per year using a count plot.
    * **Violations by Borough:** Shows the distribution of violations across different boroughs using a count plot.
    * **Critical Violation Rate Distribution:** Displays the distribution of critical violation rates using a histogram with a KDE.
    * **Violation Category by Borough:** Uses a count plot to show the frequency of different violation categories (General, Public Health Hazard, Critical) within each borough.
    * **Distribution of Violation Rates by Borough:**
        * Boxplot of `Critical Violation Rate` by `Borough`.
        * Violin plot of `Violation Rate Percent` by `Borough`.
        * Bar chart of average `Violation Rate Percent` by `Borough`.
    * **Violation Rates by Program Type:**
        * Bar chart of average `Violation Rate Percent` by `Program Type`.
        * Bar chart of average `Public Health Hazard Violation Rate` by `Program Type`.
    * **Trend of Violations Over Time:** Line chart showing the trend of average `Critical Violation Rate` and `Public Health Hazard Violation Rate` by `Inspection Year`.

4.  **Research Question Analysis & Modeling 🤖:**

    * **Q1: What is the distribution of violation rates across different boroughs?** (Addressed in EDA)
    * **Q2: How do violation rates differ between various program types (e.g., Preschool vs. Infant Toddler programs)?** (Addressed in EDA)
    * **Q3: How does the childcare center's maximum capacity influence its average number of violations per inspection?**
        * A **Linear Regression** model is trained with `Maximum Capacity` as the feature and `Violation Rate Percent` as the target.
        * Model performance is evaluated using Mean Squared Error (MSE) and R² Score.
    * **Q4: Can we predict the likelihood of a childcare center receiving critical violations based on borough, childcare type, historical violation rates, facility type, and educational worker numbers?**
        * A binary target variable `Critical Violation Received` is created (1 if `Critical Violation Rate` > 0, else 0).
        * A **Random Forest Classifier** model is trained using features: `Borough`, `Child Care Type`, `Violation Rate Percent`, `Facility Type`, and `Total Educational Workers`.
        * Categorical features are preprocessed using One-Hot Encoding.
        * Model performance is evaluated using accuracy, a classification report, and a confusion matrix.

---

## 🛠️ Libraries Used

* `google.colab.files` (for file uploads/downloads in Colab)
* `numpy`
* `pandas`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn.preprocessing.StandardScaler`
* `sklearn.preprocessing.LabelEncoder`
* `sklearn.preprocessing.OneHotEncoder`
* `sklearn.model_selection.train_test_split`
* `sklearn.model_selection.cross_val_score`
* `sklearn.linear_model.LogisticRegression` (imported but not explicitly used for a final model in the provided EDA/modeling questions)
* `sklearn.linear_model.LinearRegression`
* `sklearn.tree.DecisionTreeClassifier` (imported but not explicitly used for a final model)
* `sklearn.ensemble.RandomForestClassifier`
* `sklearn.ensemble.RandomForestRegressor`
* `sklearn.naive_bayes.GaussianNB` (imported but not explicitly used for a final model)
* `sklearn.metrics` (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error, r2_score)
* `sklearn.compose.ColumnTransformer`
* `sklearn.pipeline.Pipeline`

---

## 🚀 How to Run

1.  **Environment:** This notebook is designed to run in a Google Colab environment.
2.  **Dataset:**
    * The original dataset `DOHMH_Childcare_Center_Inspections_20250209.csv` needs to be available in the same directory as the notebook, or the path in `pd.read_csv()` (cell 2) should be updated.
    * The notebook also generates a `cleaned_child_care_compliance.csv` file which is then downloaded (cell 13). For subsequent runs or if this file is already available, the download step might be optional.
3.  **Execution:** Run the cells sequentially from top to bottom.
    * The initial cells import libraries and load the data.
    * Subsequent cells perform data cleaning, EDA visualizations, and model training/evaluation.

---

## 📊 Results & Insights (Brief)

* **EDA:** The EDA section provides various visualizations showing patterns in inspections by year and borough, distributions of different violation rates, and how violation categories differ across boroughs and program types.
* **Maximum Capacity vs. Violations (Linear Regression):**
    * The linear regression model showed a very low R² score (approx 0.0066), indicating that `Maximum Capacity` alone has a very weak linear relationship with `Violation Rate Percent`.
    * The slope was slightly negative (-0.016).
* **Predicting Critical Violations (Random Forest Classifier):**
    * The Random Forest Classifier achieved an accuracy of approximately 0.97 in predicting whether a center would receive a critical violation.
    * The classification report shows high precision and recall for identifying centers with critical violations, though the model is less effective at correctly identifying centers with *no* critical violations (lower recall for "No Critical Violation"). This suggests the model is good at catching violations but might misclassify some compliant centers.
* **Predicting Violation Rate Percent (Random Forest Regressor):**
    * When using `Maximum Capacity`, `Borough`, `Program Type`, and `Facility Type` as features, the Random Forest Regressor achieved an R² score of approximately 0.53, indicating a moderate ability to explain the variance in `Violation Rate Percent`.
## 💡 Potential Future Work

* Explore more advanced feature engineering techniques.
* Experiment with other machine learning models for both regression and classification tasks.
* Perform hyperparameter tuning for the models to potentially improve performance.
* Investigate the impact of other features not included in the current models.
* Conduct a more in-depth analysis of the "Regulation Summary" and "Health Code Sub Section" for common themes in violations.
