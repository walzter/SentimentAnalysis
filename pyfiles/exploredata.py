#!/usr/bin/env/python3

# print color text
from termcolor import colored

# linalg + df
import pandas as pd
import numpy as np 

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

def data_info(dataset_list, title=None, data_split=None, random_sample=True,visualize=True):
  """ Basic dataset information (examples, NaNs, sample, Classes)

  Parameters
  ----------
  dataset_set   : list of pandas.DataFrame
                  A list containing the loaded datasets which will be used 
  
  title         : str 
                  The title for the graph, if a different one is wanted. 

  data_split    : list of strings
                  List of strings containing the name for the desired data_split (assumes the same order as the dataframes in the list)
  
  random_sample : bool
                  Specfies whether to display a random sample from the given data_split
  
  visualize     : bool
                  Specfies whether to show the plots for the (im)balance of the datasets
  
  Returns
  -------
  Prints & shows to console a short information about the dataset. 

  """
  for idx, dataset in enumerate(dataset_list):
    # to identify which split we are looking at (just in case)
    print(colored(data_split[idx].upper(),'red'),'\n')
    # size of the dataset 
    print(f"Number of examples = {dataset.shape[0]}")
    # we can also see if there are any empty values 
    print(f"\nNumber of empty values = {dataset.isna().sum().sum()}\n") # no empty values in either column
    # print a random tweet
    if random_sample:
      rand_tweet = dataset.sample()
      label, tweet = rand_tweet['label'].values[0], rand_tweet['text'].values[0]
      print(f"Random Tweet Sample:")
      print(f"Label: {label}")
      print(f"Tweet: {tweet}\n")
    # let's visualize the class dist. 
    if visualize:
      # map the Negative/Positive to the corresponding values 0-> Negative & 1-> Positive
      g = sns.countplot(x = dataset['label'].map({
                                                  0:'Negative',
                                                  1:'Positive'
                                                  }))
      # set a title
      if title is not None:
        g.set_title(title)
      else: 
        g.set_title(f"Graph {idx+1} for {data_split[idx]} dataset: Examples per class")
      # rename the labels
      g.set_xlabel('Label / Sentiment')
      g.set_ylabel('Number of Examples')
      # show the plot 
      plt.show()
      # how many values do we have per class, in percentage, else change the normalize param
      print(dataset.value_counts('label', normalize=True)*100,'\n')
      

# Check the max sequences 
def max_seq_length(dataframe_list,tokenizer,visualize=True,return_stats=True):
    """
    
    PARAMS
    ------
    dataframe_list : list of pandas.DataFrame
                     A list containing the loaded csv files into dataframes. Assumes order of training, testing.
    
    visualize      : bool
                     Shows distribution of token graphs
    
    return_stats   : bool
                    Prints the method '.describe' of pandas.DataFrame


    RETURNS
    -------
    np.max(token) : int
                    Maximum sequence length, or the maximum number of tokens per tweet.

    """
    # we set a max length 

    # get the average length 
    test_df = pd.concat(dataframe_list)
    # number of tokens 
    tokens = [len(tokenizer.encode(x)) for x in test_df['text'].values]
    #creating the figure
    if visualize:
        g = sns.histplot(tokens)
        # titles
        g.set_title("Graph of the Number of tokens per tweet")
        plt.show()
    if return_stats:
        # some stats
        print(pd.DataFrame(tokens).describe())
    return np.max(tokens)