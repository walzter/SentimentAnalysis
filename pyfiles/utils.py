#!/usr/bin/env/python3

# system files
import os
from glob import glob
# unzipping
from zipfile import ZipFile
# dataframe 
import pandas as pd
# visualize 
import seaborn as sns
import matplotlib.pyplot as plt 
# lin alg 
import numpy as np
# dataloader
from torch.utils.data import DataLoader
# metrics
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, PrecisionRecallDisplay
# dataset we made
from dataset import SentimentDataset


#USED!! # Extract Data Helper
def extract_data(zip_file,dir_to_make=None,remove_zip=True):
  """ Extracts the content of the zipfile to the desired output 

  Parameters
  ----------
  zip_file    : path (str)
                Path to the zip file which will be decompressed 
  
  dir_to_make : path (str)
                The folder which will be made 
  
  remove_zip  : bool
                Switch for controlling if the zip file should be deleted or not. 

  Returns
  -------
  dir_to_make : path
                creates the folder, and if it already exists it will not create it. 

  zip_file    : path
                Will extract the content of the zip_file to the corresponding directory that was made (dir_to_make)

  """
  ## check if user input for dir
  if dir_to_make is not None:
    pass
  else:
    dir_to_make = './data'
  # make a new directory with chosen name to extract files
  os.makedirs(dir_to_make,exist_ok=True)
  # unzipping the file 
  with ZipFile(zip_file,'r') as zf:
    zf.extractall(dir_to_make)
    print(f"Extracted {zip_file} to {dir_to_make}")
  # to remove the zip or not to remove ?
  if remove_zip:
    os.remove(zip_file)
  else:
    pass


# Load & Remove Columns Helper 
def load_clean_data(data_dir):
  """Loads .CSV file and keeps the specified columns

  Parameter
  ----------
  data_dir : path
             The absolute or relative path where the data (training and testing file in .CSV)
  
  cols_to_use : list of integers (int)
                Index of the columns to keep, for input (X), label (Y) from the data
                Defaults to [5,0].
  
  label_names : list of strings
                Names for the input (X) and label (Y) columns of our data.
                Defaults to ['text','label']

  Returns
  -------
  test, train_df : pandas.DataFrame
                   Returns the dataframes with only positive and negative, renamed and replacing the
                   label 4 for 1.
  """


  ## only finds csv files in data_dir: test,train
  file_list = glob(data_dir+'/*.csv')

  ## reading the files
  df_list=  [pd.read_csv(df,
                            header=None,
                            usecols=[0,5],
                            names=['label','text']) for df in file_list ]
  ## now we need to replace the values 
  label_map = {
               0:0,
               4:1  
               }
  
  ## map the label to the dataframes in the list
  df_list[0]['label'], df_list[1]['label'] = df_list[0]['label'].map(label_map), df_list[1]['label'].map(label_map)
  
  ## removing the NaNs, which would be the Neutrals (2) label
  df_list = [x.dropna(axis=0) for x in df_list]
  # saving the clean dataframes
  df_list[0].to_csv('./training_clean.csv')
  df_list[1].to_csv('./testing_clean.csv')
  
  return df_list


# DataLoader Maker from dataframe
def data2loader(dataframe, tokenizer, max_length, batch_size):
    """ Converts pd.DataFrame into torch.data.DataLoader
    PARAMS
    ------
    param : type
            Description


    RETURNS
    -------
    param : type
            Description
    
    
    """   

    # making the dataset object with specified tokenizer
    data = SentimentDataset(dataframe['text'].to_numpy(),
                            dataframe['label'].to_numpy(),
                            tokenizer=tokenizer,
                            max_length=max_length
                            )
        
    # we now return the DataLoader object
    loader = DataLoader(data, batch_size=batch_size,num_workers=0)
        
    return loader