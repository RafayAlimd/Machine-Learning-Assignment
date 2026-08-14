# Road Accident Severity Prediction Using Machine Learning

## 1. Problem Statement

This project predicts road accident severity from accident-related features such as driver information, vehicle condition, road context, and time-of-day. The target variable is `Accident_severity`, with three classes: Fatal injury, Serious Injury, and Slight Injury.

The project compares five required classifiers and evaluates each model on six metrics using an untouched test set.

## 2. Dataset

Dataset: Road Traffic Accident / RTA Dataset.

- Instances: 12,316
- Original columns: 32
- Target: `Accident_severity`
- Classification type: Multiclass (3 classes)

The dataset is used for an 80/20 stratified train/test split. The test set is kept untouched during resampling.

## 3. Preprocessing

The preprocessing pipeline:

1. Removes columns identified as target-leakage variables.
2. Converts `Time` into `hour_of_day`.
3. Normalizes missing/unknown categorical values.
4. Uses ordinal encoding for ordered variables.
5. Uses one-hot encoding for nominal categorical variables.
6. Uses median imputation for numeric variables.
7. Uses `SMOTENC` on the training data only to address class imbalance.
8. Applies standard scaling before the classifiers.

The fitted preprocessing and classifier are saved together as complete pipelines so that the Streamlit app uses the same transformation process as training.

## 4. Models

The five required models are:

1. Logistic Regression
2. Decision Tree
3. KNN
4. Gaussian Naive Bayes
5. Random Forest

## 5. Streamlit App Workflow

The Streamlit app provides:

- Model selection from the sidebar
- Optional CSV download of the generated evaluation dataset
- File upload for `test_data.csv`
- Six evaluation metrics
- Confusion matrix
- Classification report
- Prediction preview
- Automatic retraining if required model artifacts are missing

The app validates the uploaded data against the saved feature metadata before scoring. The features are reordered to match the training pipeline exactly.

## 6. Evaluation Metrics

Each model is evaluated on the untouched test set using:

- Accuracy
- AUC (macro OvR)
- Precision (macro)
- Recall (macro)
- F1 (macro)
- Matthews Correlation Coefficient (MCC)

## 7. Model Comparison

| Model | Accuracy | AUC (macro OvR) | Precision (macro) | Recall (macro) | F1 (macro) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8405 | 0.5605 | 0.3899 | 0.3384 | 0.3204 | 0.0528 |
| Decision Tree | 0.7399 | 0.5716 | 0.4028 | 0.4532 | 0.4190 | 0.0713 |
| KNN | 0.8373 | 0.5158 | 0.3593 | 0.3371 | 0.3192 | 0.0289 |
| Gaussian Naive Bayes | 0.7845 | 0.4970 | 0.3263 | 0.3314 | 0.3261 | -0.0077 |
| Random Forest | 0.8458 | 0.6603 | 0.4908 | 0.3373 | 0.3148 | 0.0680 |

## 8. Model Observations

### Logistic Regression
Logistic Regression achieves strong overall accuracy, but its macro recall, macro F1, and MCC remain relatively low. This suggests that accuracy is influenced by the majority class and that the linear decision boundary does not separate all accident severities equally well.

### Decision Tree
Decision Tree achieves the highest macro recall (0.4532), macro F1 (0.4190), and MCC (0.0713) among the five models. Although its accuracy is lower than Random Forest, it demonstrates more balanced performance across the severity classes.

### KNN
KNN reaches high accuracy but lower macro recall, macro F1, and MCC. This indicates that distance-based classification is less effective in the mixed, encoded feature space for this task.

### Gaussian Naive Bayes
Gaussian Naive Bayes gives the lowest macro AUC and a slightly negative MCC. Its independence and Gaussian assumptions do not fit the complex relationships in the accident data well.

### Random Forest
Random Forest achieves the highest accuracy (0.8458), macro AUC (0.6603), and macro precision (0.4908). It also produces the strongest class-wise AUC for Fatal injury (0.6846), suggesting strong separation for the rarest severity class.

## 9. Overall Best Model

**Decision Tree** is selected as the overall best model based on balanced classification performance because it achieves the highest macro recall, macro F1, and MCC among the evaluated models.

Random Forest remains a strong alternative and achieves the highest accuracy, macro AUC, and macro precision.

Random Forest is also a strong model and achieved the highest accuracy (0.8454), macro AUC (0.6564), and macro precision (0.5465).

## 10. Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Model selection
- Six evaluation metrics
- Confusion matrix
- Classification report
- Prediction preview

Live app link: https://machine-learning-assignment-zx3srknbrmrg3ke3fccbe3.streamlit.app/

## 11. GitHub Repository

Repository link: https://github.com/RafayAlimd/Machine-Learning-Assignment/tree/main/Road_Accident_Severity_Assignment

## 12. How to Run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The training script creates the model files, `test_data.csv`, comparison CSV files, and metadata required by the Streamlit application.
