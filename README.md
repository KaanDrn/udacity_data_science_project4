# udacity_data_science_project4

# Motivation
This project shows how customer data is used from the Bertelsmann/Arvato dataset.  
The data is used to finish two main tasks, which are described below.

_Task 1 - Customer Segmentation_:
The aim of this task was to use demographic data from the population of Germany and from customer data of Bertelsmann/Arvato to find similarities and differences in the data to find possible customers for a mail order campaign.  
To achieve that, unsupervised learning methods were used. First dimension reduction methods were used and tested against each other. Afterwards, KMeans clustering algorithm was used to cluster the data. PCA in combination with KMeans performed best. It could be observed, that the data from the populations was homogeneously separated and the customers had just a few specific clusters.

_Task 2 - Mailorder Classification_:
For the second task, we got data from a email campaign which was already performed and where we already got responses of the customers. The data was additionally enriched with the cluster-labels from the trained clustering pipeline trained in part 1.  
A SVM-classifier was used to classify the data. The data was highly imbalanced, which was also a challenge which was solved with upsampling the data.  
The F1-Score on the training data was 0.7943.

_Table 1: Confusion matrix for the classification_
||P|N|
|-|-|-|
|P|353|9900|
|N|71|23890||

# Libraries
The analysis makes use of the most known python libraries, namely:
- `pandas`
- `numpy`
- `matplotlib`
- `sklearn`

In addition `pickle` is used to save the results from the classification task 2.

# Files
The repo contains one directory (`data`), where the results of the classification and the data from udacity are stored. The udacity data should not be checked in, since the data has a high volume.

The code and the report is documented in the `Arvato Project Workbook.ipynb`

The two `DIAS*.xlsx` files, describe the features, as well as the meaning of their values.

The `data/RESULT_mailout_test.pkl` file is a pickle-file, where the classification of the test dataset is stored.

# Summary
A clustering pipeline was implemented, which consists of a dimension reduction step and clustering step. The 
 data was then enriched by the clusters, that the data was separated in and a classification pipeline was train with that data. Then the performance of the data was tested on the train data set and was optimized using cross validation and grid-search.
 The results of the labels on the Test set are stored in `data/RESULT_mailout_test.pkl`.

# Acknowledgments
Thanks to Bertelsmann and Arvato for providing this huge dataset and giving udacity students the opportunity to experience such an interesting, real life use-case of customer segmentation.