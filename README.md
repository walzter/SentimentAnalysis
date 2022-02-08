## Project doing Sentiment Analysis (WIP) 

## Content 
The project consists of taking a list of tweets with an associated emotion (0 for negative and 1 for positive) and then classifying it using a NLP model. 

- Dataset (Train & Test) --> Watch out the training is imbalanced ! 
- Jupyter Notebook running through the process 
- pyfiles -> all the scripts for preprocessing, EDA, model creation, training & evaluation 
- TODO file which inlcudes future ways to improve the model 


## Steps Done: 
- Extract Zip file 
- Load Data 
- EDA 
- Get maximum sequence length 
- Create a model (in this case a classifier on top of pre-trained BERT) 
- Training with verbosity for steps
- Testing with evaluation metrics (Confusion Matrix, ROC, PR) 
- Final testing accuracy 
