## Function Name: cross_validation()

### Description:
This function performs k-fold cross-validation on a given machine learning model using the input features (X) and target variable (y) provided. It returns the mean score of the model's performance across all the folds.

### Parameters:

X: Input features (data matrix) of shape (n_samples, n_features).
y: Target variable (label vector) of shape (n_samples,).
model_clf: Machine learning model object that implements the sklearn API, such as a classifier or regressor.
Returns:

model_clf_score.mean(): Mean score of the model's performance across all the folds.
Dependencies:

This function requires the following modules from the scikit-learn library: cross_val_score.
It also uses the 'n_jobs' parameter in the cross_val_score function, which allows for parallel computation of cross-validation folds.

## Function Name: acc_rec_f1_roc_aug_score()

### Description:
This function calculates and prints the accuracy, recall, F1 score, and ROC AUC score of a given machine learning model's predictions using the input features (X) and target variable (y) provided.

### Parameters:

X: Input features (data matrix) of shape (n_samples, n_features).
y: Target variable (label vector) of shape (n_samples,).
model_clf: Machine learning model object that implements the sklearn API, such as a classifier or regressor.
Dependencies:

This function requires the following modules from the scikit-learn library: accuracy_score, recall_score, f1_score, and roc_auc_score.