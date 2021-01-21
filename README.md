# Stanza detection evaluation

Repository with all the information to replicate our results


Corpus dataset for evaluation on rantanplan includes poems under copyright so we are not able to publish it online but we can share it privately in order to replicate our results

# Dependency installation

`pip install -r requirements.txt`

## rantanplan

Open the notebook rantanplan `Evaluation.ipynb` and execute the cells in order. Since rantanplan uses no training set, the whole corpus is evaluated.

## Glove+LSTM

Open the notebook rantanplan `Glove+LSTM Evaluation.ipynb` and execute the cells in order. The weights of our trained model are downloaded and then loaded into the model that will be used on the test dataset.

## BERT+LSTM

Open the notebook rantanplan `BERT+LSTM Evaluation.ipynb` and execute the cells in order. The weights of our trained model are downloaded and then loaded into the model that will be used on the test dataset. This evaluation may need up to 32GB.
