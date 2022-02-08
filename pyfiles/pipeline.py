#!/usr/bin/env/python3

# garbage collect 
import gc

# import transformer tokenizer + warmup scheduler
from transformers import BertTokenizer,get_linear_schedule_with_warmup
# torch 
import torch
# optimizer
from torch.optim import AdamW
# loss function
from torch.nn import CrossEntropyLoss

# utils
from utils import load_clean_data, data2loader
# our model 
from model import BertFeels


def model_pipeline(config, eda=True):
  # Step 1:  extract the data locally
  #extract_data(config_dict['DATA'],dir_to_make=config_dict['DATA_DIR'], remove_zip=False)

  # Step 2: load the files and remove the undesired columns (Pre-Processing)
  df_list = load_clean_data(config['DATA_DIR'])

  # Step 3: We set the tokenizer
  tokenizer = BertTokenizer.from_pretrained(config['PRE_TRAINED_MODEL'])

  # Step 4: Add a Validation step just in case (from the training set)
  # train & val
  # testing
  df_train, df_test = df_list

  # Step 5: Generate the DataLoaders 
  train_loader = data2loader(df_train, tokenizer, config['MAX_LENGTH'], config['BATCH_SIZE'])
  test_loader  = data2loader(df_test, tokenizer, config['MAX_LENGTH'], config['BATCH_SIZE'])

  # Step 6: instantiate model & send to device
  model = BertFeels(config['NUM_CLASSES'])
  model = model.to(config['DEVICE'])

  # Step 7: Optimizer, Scheduler & Loss Function (criteria)
  # optimizer
  optimizer = AdamW(model.parameters(), lr=config['LEARNING_RATE'])
  # schedule
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=len(train_loader) * config['EPOCHS'])
  # loss function
  criterion = nn.CrossEntropyLoss().to(config['DEVICE'])
  # OPTIONAL: Clean cache & garbage
  torch.cuda.empty_cache()
  gc.collect()


  return model, optimizer, scheduler, criterion, train_loader, test_loader,df_train,df_test