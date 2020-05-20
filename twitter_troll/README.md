# Twitter Troll Classifier
## Datasets
There were two datasets utilized in this analysis - a troll tweet dataset and a control political tweet dataset.

Troll tweet dataset - https://github.com/fivethirtyeight/russian-troll-tweets. With thanks to FiveThirtyEight.

Control political tweet dataset - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/AEZPLU&version=1.0. With thanks to Harvard University researchers.

## Notebook

`troll_tweet_classifier.ipynb` contains the code used for cleaning the data, generating BERT word embeddings on Tweet content and training and testing the logistic regression classifier. Some of the data was cleaned prior to this step.

## Results

This classifier was able to achieve 95%+ accuracy in this task, which is in line with results in literature using similar techniques.
