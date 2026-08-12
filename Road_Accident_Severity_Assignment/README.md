# Road Accident Severity Prediction Using Machine Learning

## 1. Problem Statement

This project predicts road accident severity using machine-learning classification models. The target variable is `Accident_severity`, with three classes: Fatal injury, Serious Injury, and Slight Injury.

The project compares five required classifiers and evaluates them using six metrics.

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

## 5. Evaluation Metrics

Each model is evaluated on the untouched test set using:

- Accuracy
- AUC (macro OvR)
- Precision (macro)
- Recall (macro)
- F1 (macro)
- Matthews Correlation Coefficient (MCC)

## 6. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8401 | 0.5609 | 0.3862 | 0.3383 | 0.3203 | 0.0506 |
| Decision Tree | 0.7435 | 0.5859 | 0.4112 | 0.4673 | 0.4280 | 0.1085 |
| KNN | 0.8364 | 0.5160 | 0.3506 | 0.3360 | 0.3173 | 0.0199 |
| Gaussian Naive Bayes | 0.7845 | 0.4967 | 0.3263 | 0.3314 | 0.3261 | -0.0077 |
| Random Forest | 0.8454 | 0.6564 | 0.5465 | 0.3477 | 0.3328 | 0.0715 |

## 7. Model Observations

### Logistic Regression
Logistic Regression achieved high overall accuracy but relatively low macro F1 and MCC. This indicates that accuracy is influenced strongly by the majority class and that the linear decision boundary does not separate all severity classes equally well.

### Decision Tree
Decision Tree produced the highest macro recall, macro F1, and MCC among the five models. Although its accuracy was lower than Random Forest, its stronger macro scores indicate more balanced performance across the severity classes.

### KNN
KNN achieved high accuracy but comparatively low macro recall, macro F1, and MCC. This suggests that distance-based classification is less effective for the mixed, high-dimensional encoded feature space.

### Gaussian Naive Bayes
Gaussian Naive Bayes produced the lowest macro AUC and a slightly negative MCC. Its conditional-independence and Gaussian assumptions do not fit the complex relationships in the accident data particularly well.

### Random Forest
Random Forest achieved the highest accuracy, macro AUC, and macro precision. It also produced the strongest Fatal Injury one-vs-rest AUC (0.6820), showing useful separation of the rarest severity class.

## 8. Overall Best Model

**Decision Tree** is selected as the overall best model based on balanced classification performance because it achieved the highest macro recall (0.4673), macro F1 (0.4280), and MCC (0.1085).

Random Forest is also a strong model and achieved the highest accuracy (0.8454), macro AUC (0.6564), and macro precision (0.5465).

## 9. Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Model selection
- Six evaluation metrics
- Confusion matrix
- Classification report
- Prediction preview

Live app link: **ADD YOUR STREAMLIT LINK HERE**

## 10. GitHub Repository

Repository link: **ADD YOUR GITHUB LINK HERE**

## 11. How to Run

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The training script creates the model files, `test_data.csv`, comparison CSV files, and metadata required by the Streamlit application.

## 12. Submission Links

- GitHub Repository: https://github.com/RafayAlimd/Machine-Learning-Assignment/tree/main/Road_Accident_Severity_Assignment
- Live Streamlit App: **ADD LINK**
- BITS Virtual Lab Screenshot: **ADD TO FINAL PDF**
