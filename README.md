# Titanic - Machine Learning from Disaster

An end-to-end Machine Learning pipeline developed on Kaggle to predict passenger survival on the Titanic.

## Project Overview
- **Objective:** Predict whether a passenger survived the Titanic shipwreck based on demographic and trip data.
- **Dataset:** Kaggle Titanic Competition dataset (`train.csv`, `test.csv`).

## Workflow & Methodology
1. **Exploratory Data Analysis (EDA):** Analyzed missing values (`Age`, `Cabin`, `Embarked`) and dataset distributions.
2. **Feature Engineering:** 
   - Grouped `Age` into categorical bins (`AgeGroup`).
   - Created `FamilySize` combining `SibSp` and `Parch`.
   - Extracted passenger titles (`Title`) from names.
3. **Preprocessing:** Handled missing values using medians/modes and applied one-hot encoding to categorical features.
4. **Model Tuning & Selection:** Trained and evaluated multiple classifiers via `GridSearchCV`:
   - Logistic Regression
   - Random Forest (Selected best performer)
   - Support Vector Classifier (SVC)
   - K-Nearest Neighbors (KNN)
   - XGBoost

## Results
- **Validation Accuracy:** ~83.2% (Random Forest Classifier)
- **Top Validation Metrics (Random Forest):** Precision: 0.8235 | Recall: 0.7568 | F1-Score: 0.7887
