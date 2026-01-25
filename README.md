# Heart Disease Prediction using Machine Learning

## a. Problem Statement
The objective of this project is to predict the presence of heart disease in patients using machine learning classification techniques. Multiple supervised learning models are trained and evaluated on a publicly available heart disease dataset. The models are compared using standard evaluation metrics, and a Streamlit web application is developed to allow interactive dataset upload, model selection, and visualization of results.

## b. Dataset Description
The Heart Disease dataset was obtained from a public repository (UCI / Kaggle).

Dataset Characteristics:
- Number of instances: 1025  
- Number of features: 13 input features + 1 target  
- Type: Binary classification (0 = No Heart Disease, 1 = Heart Disease)

Features:
- age – Age of patient  
- sex – Gender (0 = Female, 1 = Male)  
- cp – Chest pain type  
- trestbps – Resting blood pressure  
- chol – Serum cholesterol  
- fbs – Fasting blood sugar  
- restecg – Resting ECG  
- thalach – Max heart rate  
- exang – Exercise induced angina  
- oldpeak – ST depression  
- slope – Slope of ST segment  
- ca – Number of major vessels  
- thal – Thalassemia  
- target – Heart disease (0/1)

## c. Models Used and Evaluation Metrics
### Comparison Table
Matthews Correlation Coefficient (MCC)
| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.8488   | 0.9171 | 0.8306    | 0.9115 | 0.8692 | 0.6951 |
| Decision Tree       | 0.9024   | 0.9491 | 0.8780    | 0.9558 | 0.9153 | 0.8048 |
| KNN                 | 0.8195   | 0.9364 | 0.8393    | 0.8319 | 0.8356 | 0.6356 |
| Naive Bayes         | 0.8488   | 0.9060 | 0.8475    | 0.8850 | 0.8658 | 0.6937 |
| Random Forest       | 0.9512   | 0.9902 | 0.9558    | 0.9558 | 0.9558 | 0.9014 |
| XGBoost             | 0.9415   | 0.9834 | 0.9316    | 0.9646 | 0.9478 | 0.8819 |
## Model Observations
| ML Model                 | Observation                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Provides stable performance with good recall and balanced metrics, suitable as a baseline model.              |
| Decision Tree            | Achieves strong accuracy but shows tendency toward overfitting due to model complexity.                       |
| KNN                      | Performance depends on feature scaling and choice of K; slightly lower accuracy compared to ensemble methods. |
| Naive Bayes              | Fast and simple model but assumes feature independence, leading to slightly reduced performance.              |
| Random Forest (Ensemble) | Achieves the highest accuracy and MCC, showing excellent generalization due to ensemble averaging.            |
| XGBoost (Ensemble)       | Provides near-optimal performance with high AUC and F1 score, demonstrating powerful boosting capability.     |

