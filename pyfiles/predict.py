#!/usr/bin/env/python3

# torch 
import torch
import torch.nn.functional as F

from collections import defaultdict
import numpy as np
# config file
from config import make_config
# utils 
from plots import display_metrics
#from wandbtrack import log_plots_wandb

config_dict = make_config()
# getting the predictions for a classification report 
def make_preds(model, loader):
  """

  """
  # into eval mode
  model = model.eval()
  # prediction, probabilities & true values
  y_preds = []
  probas = []
  y = []

  with torch.no_grad():
    for data in loader:
      inputs = data['input_ids'].to(config_dict['DEVICE'])
      attention_mask = data['attention_mask'].to(config_dict['DEVICE'])
      labels = data['label'].to(config_dict['DEVICE'])
      
      # now getting the outputs 
      out = model(inputs, attention_mask)
      # predicted probs
      proba = F.softmax(out, dim=1).detach().cpu().numpy().tolist()
      probas.extend(proba)
      
      #predictions & loss
      _, y_pred = torch.max(out, dim=1)
      
      # extending the lists 
      y_preds.extend(y_pred)
      # same for actual values 
      y.extend(labels)

  # return the stacked lists
  y_preds = torch.stack(y_preds).cpu()
  y = torch.stack(y).cpu()

  return y_preds, y, probas


# the final predictions 
def make_final_predictions(history, model, test_loader, criteria):

  # making the predictions 
  y_preds, y, probas = make_preds(model, test_loader)
  # adjusting 
  y_preds_ls = y_preds.numpy().tolist()
  y_ls = y.numpy().tolist()
  # class names
  class_names = ['Negative','Positive']
  # logging the custom plots 
  #log_plots_wandb(y_ls, y_preds_ls, probas, class_names)
  # displaying the metrics 
  display_metrics(history, y, y_preds, class_names)