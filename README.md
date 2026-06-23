## Hotel Booking Cancellation Prediction
###**Project Overview**
This project focuses on predicting hotel booking cancellations using a dataset of hotel booking demand. The goal is to develop and evaluate various machine learning models to accurately forecast cancellations, which can help hotels optimize their operations and revenue management.

**Data**
The dataset used in this project is sourced from Kaggle and contains detailed information about hotel bookings, including customer demographics, booking specifics, and historical cancellation patterns.

**Installation**
To run this notebook, you'll need the following libraries. You can install them using pip:

pip install pandas numpy tensorflow scikit-learn matplotlib seaborn kagglehub joblib
Usage
Clone the repository or download the notebook.
Run all cells in the notebook. The notebook covers:
Data Loading and Initial Exploration
Data Preprocessing (Handling missing values, encoding categorical features, feature scaling)
Data Splitting (Training, Validation, and Test sets)
Model Training and Evaluation (Logistic Regression, Decision Tree, Random Forest, Neural Networks)
Visualization of Results (ROC Curves, Confusion Matrices, Learning Curves)
Saving the Best Model
**Model Evaluation**
Several models were trained and evaluated on metrics such as Accuracy, Precision, Recall, and F1-Score. The results DataFrame in the notebook provides a comparative overview of each model's performance.

Model	Accuracy	Precision	Recall	F1-Score	Hyperparameters	Observations
exp1:Logistic Regression	0.7708	0.6523	0.3560	0.4606	max_iter=1000	Basic linear model, serves as a performance baseline.
exp2:Decision Tree	0.7963	0.6284	0.6340	0.6312	max_depth=None, min_samples_leaf=1	Performs well, particularly in recall, suggesting good capture of positive cases. Simpler than Random Forest.
exp3:Random Forest Baseline	0.8180	0.7809	0.4698	0.5866	n_estimators=100, max_depth=10	Significant improvement over Logistic Regression, good precision.
exp4:Random Forest Tuned	0.8377	0.7707	0.5830	0.6638	n_estimators=150, max_depth=15, min_samples_split=5	Best F1-Score among tree-based models, improved recall with tuning.
exp5:NN Simple	0.8230	0.7274	0.5694	0.6388	2 Dense layers (64, 32 neurons)	Solid performance, comparable to RF baseline in F1.
exp6: Deeper NN with Dropout	0.8296	0.7343	0.5957	0.6578	3 Dense layers (128, 64, 32 neurons) with 2 Dropout(0.3) layers	Deeper architecture with dropout improved generalization and recall over simple NN.
exp7:NN Batch Norm	0.8297	0.7153	0.6324	0.6713	3 Dense layers with Batch Norm, 2 Dropout(0.3) layers, EarlyStopping(patience=5)	Achieved the highest recall among NN models, Batch Norm aided stability and performance.
The Random Forest Tuned model (exp4) achieved the highest F1-Score, making it the selected best model for this project.
