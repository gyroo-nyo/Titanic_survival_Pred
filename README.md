# 🚢 Titanic Survival Prediction

This project uses **Random Forest Classifier** to predict whether a passenger survived the Titanic disaster based on selected features from the Titanic dataset.  
It demonstrates **data preprocessing, model training, hyperparameter tuning with GridSearchCV, evaluation, and model saving/loading**.

---

## 📂 Project Overview

1. **Dataset Used**: `Titanic-Dataset.csv`
2. **Algorithm**: Random Forest Classifier
3. **Main Steps**:
   - Handle missing values with `SimpleImputer`
   - Encode categorical features (Sex: male → 0, female → 1)
   - Train/test split of the dataset
   - Train and evaluate the model
   - Perform **cross-validation**
   - Tune hyperparameters using **GridSearchCV**
   - Generate **confusion matrix** and classification report
   - Save the trained model using `joblib`

---

## ⚙️ Requirements

Install the required dependencies before running the code:

```bash
pip install pandas numpy scikit-learn matplotlib joblib


📊 How It Works
Data Preprocessing

Missing values are filled with the most frequent value for each column.

Categorical data in the Sex column is mapped to numeric values.

Model Training

Uses RandomForestClassifier from scikit-learn.

Trains on features: Pclass, Sex, Fare.

Evaluation

Accuracy Score

Cross-Validation

Confusion Matrix

Classification Report

Hyperparameter Tuning

n_estimators: [50, 100, 200]

max_depth: [None, 10, 20]

min_samples_split: [2, 5, 10]

Model Saving & Loading

Model is saved as titanic_survival.joblib for reuse.


▶️ How to Run
Place Titanic-Dataset.csv in the project directory.

Run the Python script:

bash
Copy
Edit
python titanic_model.py
The script will:

Train and evaluate the model

Show the confusion matrix

Save the trained model

📈 Example Output
Confusion Matrix:
The confusion matrix visually shows the number of correct and incorrect predictions.

Classification Report:

markdown
Copy
Edit
              precision    recall  f1-score   support

Not Survived       0.xx      0.xx      0.xx        xx
    Survived       0.xx      0.xx      0.xx        xx

    accuracy                           0.xx       xxx
   macro avg       0.xx      0.xx      0.xx       xxx
weighted avg       0.xx      0.xx      0.xx       xxx
📦 Files
titanic_model.py → Main training & evaluation script

Titanic-Dataset.csv → Dataset file (not included in repo by default)

titanic_survival.joblib → Saved trained model

🏆 Results
Accuracy and performance metrics vary depending on the dataset split.

Hyperparameter tuning improves the generalization performance.

💡 Future Improvements
Use additional features from the dataset.

Try other classifiers (e.g., Logistic Regression, XGBoost).

Deploy the model as a web app using Flask/Streamlit.

📈 Example Output
Confusion Matrix:
The confusion matrix visually shows the number of correct and incorrect predictions.

Classification Report:

markdown
Copy
Edit
              precision    recall  f1-score   support

Not Survived       0.xx      0.xx      0.xx        xx
    Survived       0.xx      0.xx      0.xx        xx

    accuracy                           0.xx       xxx
   macro avg       0.xx      0.xx      0.xx       xxx
weighted avg       0.xx      0.xx      0.xx       xxx

💡 Future Improvements
Use additional features from the dataset.

Try other classifiers (e.g., Logistic Regression, XGBoost).

Deploy the model as a web app using Flask/Streamlit.

📜 License
This project is for educational purposes only

