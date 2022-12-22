# EPFL CS-433: PROJECT 2: Enzyme-Stability-Prediction


We participated in a Kaggle Competition, asking to develop models that can predict the ranking of protein thermostability (as measured by melting point, tm) after single-point amino acid mutation and deletion.

Link to the competition : https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/





## Code description 


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


<ol>
  <li>Convolutional Neural Networks experiments</li>
  <li>Data engineeering</li>
  <li>Language model : Protbert</li>
</ol>




### `notebooks/experiments`

##### '1-Conv1d_OneChannel.ipynb' 

CNN on Amino Acid Sequence (AAS) using integer encoding on the 20 amino acid letters producing one channel array for each sequence

##### '2-One_hot_encoding.ipynb'
CNN on Amino Acid Sequence (AAS) using one-hot encoding on the 20 amino acid letters producing 20  channels array for each sequence
##### '3-Less_Channels.ipynb'
Grouping  amino acids with similar properties into 4 groups and producing a 4- channel array for each sequence 


##### '4-Test_Model_DeepSF.ipynb'
##### '5-AlexNet.ipynb'
For each of these experiments we use different CNN architectures and compare with our designed model

##### '6-CNN_optimization.ipynb'
Further optimization of our designed model 


##### '10-ProtBert_Mutation.ipynb'
Pretrained Language model prediction on protein stability.
ProtBert is already pretrained on millions of protein sequences, we adapt it to our problem


##### 'proteins_groups.ipynb'
##### 'helpers.py'

### `notebooks/exploratory`
2 notebooks for data exploration and training data correction

### `ressources`

ressources (articles, papers) that inspired our methods



---
## Authors 

* Hichem Hadhri
* Mathilde Morelli
* Iris Toye


