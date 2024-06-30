# IS5126_Project - Loan Default Prediction

## Team Members
- Chook Win Yan (A0290547X) - [e1326000@u.nus.edu](mailto:e1326000@u.nus.edu)
- Lalitha Ravi (A0268254X) - [e1101553@u.nus.edu](mailto:e1101553@u.nus.edu)
- Sreelakshmi (A0268357N) - [e1101656@u.nus.edu](mailto:e1101656@u.nus.edu)

## Institution
National University of Singapore, Singapore
## Project Overview
This project aims to develop a machine learning model to predict loan defaults. By leveraging historical loan data, our model identifies high-risk profiles, empowering banks to make informed lending decisions and promoting financial stability.

Within the loan approval process, a borrower who fails to
fulfill their repayment obligations is considered a default. This
leads to financial losses for the bank. To proactively manage
this risk and enhance lending effectiveness, we propose developing a machine learning model. This model will leverage
historical loan data and be optimized through experimentation
with various machine-learning techniques. Its primary objective-
tive is to accurately predict the likelihood of loan defaults
for individual borrowers. By identifying high-risk profiles, the
model empowers banks to make informed lending decisions,
ultimately promoting financial stability.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `tensorflow`

  
### Dataset
The dataset is sourced from Kaggle through Lending Club and comprises 396,030 records with 15 string columns and 12 numerical columns. The class distribution indicates that 19.24% of loans end in default, while 80.76% are fully paid.

### Data Preprocessing
1. Standardize columns with multiple data types.
2. Handle null values by dropping records with missing data.
3. Eliminate redundant columns to prevent multicollinearity.
4. Introduce a new category `self-employed` for empty `emp_title` values.
5. Standardize `employment length` values.
6. Merge infrequently occurring `home_ownership` categories.
7. Discard the `loan application title` column.
8. Ensure zero null values in the cleaned dataset.


### Methodology
#### Class Imbalance Handling
1. **Random Undersampling**: Integrate undersampling into each fold of the hyperparameter tuning process using stratified k-fold cross-validation.
2. **Model-Specific Techniques**:
   - Logistic Regression: Use the `class_weight` parameter.
   - XGBoost: Utilize `scale_pos_weight`.
   - Random Forest: Apply `Balanced` class weight balancing techniques.

#### Experimental Setup
1. **Experiment 1**: Training on the entire dataset without hyperparameter adjustments.
2. **Experiment 2**: Training with hyperparameter tuning using grid search and randomized search.
3. **Experiment 3**: Training with hyperparameter tuning and undersampling.

### Models Used
1. **Naïve Bayes**: Gaussian Naïve Bayes with hyperparameter tuning using `RandomizedSearchCV`.
2. **Random Forest**: Hyperparameter tuning with `RandomizedSearchCV`.
3. **Logistic Regression**: Using Stochastic Gradient Descent (SGD) Classifier with scaling.
4. **XGBoost**: Hyperparameter tuning with `RandomizedSearchCV`.
5. **Artificial Neural Network (ANN)**: Structured as a sequential model with ReLU activation and dropout layers.

### Results
The best-performing model was the optimized XGBoost model. For fully paid loans, the model achieved a precision of 0.94, a recall of 0.80, and an F1-score of 0.87. For charged-off loans, the model achieved a precision of 0.50, a recall of 0.80, and an F1-score of 0.62. The Precision-Recall Curve (PRC) for the model was 0.78, and the Receiver Operating Characteristic (ROC) curve area was 0.91. These results indicate that the model is highly effective in predicting loan defaults, with strong performance metrics across various evaluation criteria.

### Contributors
- Chook Win Yan
- Lalitha Ravi
- Sreelakshmi

For any questions or issues, please contact us at our respective email addresses.

