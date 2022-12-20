# EPFL CS-433: PROJECT 2: Enzyme-Stability-Prediction


a kaggle competition, to develop models that can predict the ranking of protein thermostability (as measured by melting point, tm) after single-point amino acid mutation and deletion.


The work was mainly done in 2 ways:

Preprocessing can include different combinations of the following methods: (1)  (2)  (3) 

Then CNN

The entire project uses different python libraries

Please add the files train.csv and test.csv directly in the repository.

## Code description 

### `run.py`


---

### `implementations.py`

This file contains the required functions as stated in the project outline pdf file.

* *mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression*
* *logistic_regression, reg_logistic_regression*

As well as auxiliary functions supporting the ones cited above.

* *compute_mse_loss, compute_mse_gradient, batch_iter, compute_stoch_mse_gradient, sigmoid, calculate_logistic_loss, calculate_logistic_gradient*
* *calculate_stoch_logistic_gradient, stoch_reg_logistic_regression*

---

### `data_processing.py`


--- 

### `our_progress_run.ipynb`

A notebook outlining the step-by-step progress of the model (each stage adds something on top of the previous version):

1. logistic regression 
2. logistic regression + normalized 
3. logistic regression + normalized + w0
4. logistic regression + normalized smart + w0
5. logistic regression + normalized smart + w0 + high correlation features removed


---
### `seven_methods.py`

This file allows to calculate the accuracy for seven methods of regression and classification coded for this project.

* A. Gradient Descent with MSE
* B. Stochastic Gradient Descent with MSE
* C. Least Squares 
* D. Ridge Regression with cross validation to find best lambda
* E. Logistic Regression with cross validation to find best lambda
* F. Regularized Logistic Regression
* G. K-nearest neighbors classification



---
## Authors 

* Hichem Hadhri
* Mathilde Morelli
* Iris Toye


