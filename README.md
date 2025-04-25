# CancerTree-Decision-Tree-Classifier-for-Breast-Cancer-Diagnosis
CancerTree is a Python-based machine learning project that uses a Decision Tree Classifier to predict the presence of breast cancer based on the popular Wisconsin Breast Cancer Dataset. The project also explores model complexity by evaluating tree depth and performs hyperparameter tuning using GridSearchCV. The dataset used is the Breast Cancer Wisconsin Dataset, available via sklearn.datasets. It includes 569 instances described by 30 numeric features, along with binary target labels (malignant or benign).

# Features
 - Decision Tree Classifier using the entropy criterion.
 - Holdout Validation with a 60/40 stratified train-test split.
 - Accuracy Evaluation using sklearn.metrics.accuracy_score.
 - Tree Visualization via matplotlib and plot_tree.
 - Model Complexity Analysis: plots training/test accuracy vs. max tree depth.
 - Hyperparameter Tuning with GridSearchCV.

