This project builds and compares six machine learning classifiers—Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost—to predict target classes from input features. Models are evaluated using standard metrics, and a Streamlit app enables interactive data upload, model selection, and result visualization.
| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.8488   | 0.9171 | 0.8306    | 0.9115 | 0.8692 | 0.6951 |
| Decision Tree       | 0.9024   | 0.9491 | 0.8780    | 0.9558 | 0.9153 | 0.8048 |
| KNN                 | 0.8195   | 0.9364 | 0.8393    | 0.8319 | 0.8356 | 0.6356 |
| Naive Bayes         | 0.8488   | 0.9060 | 0.8475    | 0.8850 | 0.8658 | 0.6937 |
| Random Forest       | 0.9512   | 0.9902 | 0.9558    | 0.9558 | 0.9558 | 0.9014 |
| XGBoost             | 0.9415   | 0.9834 | 0.9316    | 0.9646 | 0.9478 | 0.8819 |

| ML Model | Observation about Model Performance |
|---------|-----------------------------------|
| Logistic Regression | Provided baseline performance with good recall but comparatively lower accuracy than ensemble models. |
| Decision Tree | Achieved high recall but showed signs of overfitting due to its simple tree structure. |
| KNN | Delivered moderate accuracy and was sensitive to feature scaling and data distribution. |
| Naive Bayes | Produced consistent results but lower accuracy due to its strong independence assumptions. |
| Random Forest | Achieved the best overall performance with high accuracy and MCC by combining multiple decision trees. |
| XGBoost | Demonstrated strong generalization with high accuracy and AUC, performing close to Random Forest. |
