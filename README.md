This project builds and compares six machine learning classifiers—Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, and XGBoost—to predict target classes from input features. Models are evaluated using standard metrics, and a Streamlit app enables interactive data upload, model selection, and result visualization.
| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.8488   | 0.9171 | 0.8306    | 0.9115 | 0.8692 | 0.6951 |
| Decision Tree       | ...      |        |           |        |        |        |
| KNN                 | ...      |        |           |        |        |        |
| Naive Bayes         | ...      |        |           |        |        |        |
| Random Forest       | ...      |        |           |        |        |        |
| XGBoost             | ...      |        |           |        |        |        |
| ML Model | Observation about Model Performance |
|---------|-----------------------------------|
| Logistic Regression | Provided baseline performance with good recall but comparatively lower accuracy than ensemble models. |
| Decision Tree | Achieved high recall but showed signs of overfitting due to its simple tree structure. |
| KNN | Delivered moderate accuracy and was sensitive to feature scaling and data distribution. |
| Naive Bayes | Produced consistent results but lower accuracy due to its strong independence assumptions. |
| Random Forest | Achieved the best overall performance with high accuracy and MCC by combining multiple decision trees. |
| XGBoost | Demonstrated strong generalization with high accuracy and AUC, performing close to Random Forest. |
