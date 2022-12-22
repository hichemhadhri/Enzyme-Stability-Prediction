# EPFL CS-433: PROJECT 2: Enzyme-Stability-Prediction


We participated in  a Kaggle Competition, asking to develop models that can predict the ranking of protein thermostability (as measured by melting point, tm) after single-point amino acid mutation and deletion.

Link to the competition : https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/


The work was mainly done in 2 ways:
Preprocessing can include different combinations of the following methods: (1)  (2)  (3) 
Then CNN
The entire project uses different python libraries: numpy, pandas, scipy, 


## Code description 

---

### `data`

This folder contains all the csv files that were given or added by us:
* clean_train_data.csv (obtained from ?)
* test.csv (original test set)
* train.csv (original training set)
* train_updates_20220929.csv  (correction for training set **usage explained** in `experiments/models/exploratory/data_correction.ipynb`)
* train_v1.csv (result of correcting dataset)



---
## `Experiments`
Each experiment is in a self-contained notebook and can be run to reproduce the same results as in the report.

<br> The experiments can be divided into 3 different groups

<li>
    <ul>1. Convolutional Neural Networks</ul>
    <ul>2. Data Grouping </ul>
    <ul>3. Language model : RosLab/ProtBert</ul>

</li>


### `notebooks/experiments`

##### `1-Conv1d_OneChannel.ipynb'`

CNN on Amino Acid Sequence (AAS) using integer encoding on the 20 amino acid letters producing one channel array for each sequence

##### `2-One_hot_encoding.ipynb`
CNN on Amino Acid Sequence (AAS) using one-hot encoding on the 20 amino acid letters producing 20  channels array for each sequence
##### `3-Less_Channels.ipynb`
Grouping  amino acids with similar properties into 4 groups and producing a 4- channel array for each sequence 


##### `4-Test_Model_DeepSF.ipynb`
##### `5-AlexNet.ipynb`
For each of these experiments we use different CNN architectures and compare with our designed model

##### `6-CNN_optimization.ipynb`
Further optimization of our designed model 


##### `7-Effect_mutations_alone.ipynb`
?
##### `8-Effect_mutations-xgboostl.ipynb`
?

##### `9-Effect_mutations_and_seqs.ipynb`
?

##### `10-Protbert.ipynb`
use protBert to predict thermostability of a sequence


##### `11-Protbert_Mutation.ipynb`
use pairs of mutant and wild type sequences to train protBert and predict difeerence in thermostability

<br>
**Note:** Both notebooks have been ran on google Colab  and their results are saved in the `experiments/models/protbert` folder

##### `helpers.py`
contain common functions used in the experiments

--- 


#### Note :  The plots, logs , and saved models  are in their respective sub folders in `experiments`
### `notebooks/experiments/data_exploration`
a small discovery notebook to explore the data and understand it better


# Requirements
pytorch 
transformers 
tokenizers
numpy
pandas
scipy
sklearn
matplotlib
seaborn
xgboost
tqdm




---
## Authors 

* Hichem Hadhri
* Mathilde Morelli
* Iris Toye


