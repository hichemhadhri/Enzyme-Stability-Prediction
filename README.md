# EPFL CS-433: PROJECT 2: Enzyme-Stability-Prediction


We participated to a Kaggle Competition, asking to develop models that can predict the ranking of protein thermostability (as measured by melting point, tm) after single-point amino acid mutation and deletion.


The work was mainly done in 2 ways:
Preprocessing can include different combinations of the following methods: (1)  (2)  (3) 
Then CNN
The entire project uses different python libraries: numpy, pandas, scipy, 


## Code description 

### `run.py`

This file produces the same file used to obtain the team's best score on the Kaggle competition. It is self-contained and only requires access to the data and files described below.

---

### `data`

This folder contains all the csv files that were given or added by us:
* clean_train_data.csv
* test.csv
* train.csv
* train_updates_20220929.csv
* train_v1.csv


---

### `models`

#### `protC1D.py`

This is the main model we used.

---

### `notebooks/experiments`

##### '1-Conv1d_OneChannel.ipynb'
##### '1-Conv1d_OneHot-Loss.ipynb'
##### '2-One_hot_encoding.ipynb'
##### '3-Less_Channels.ipynb'
##### '4-Test_Model_DeepSF.ipynb'
##### '5-AlexNet.ipynb'
##### '7-ProtBert.ipynb'
##### '8-ProtBert+LGBM.ipynb'

##### 'proteins_groups.ipynb'
##### 'helpers.py'

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


