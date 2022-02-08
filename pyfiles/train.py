#!/usr/bin/env/python3

# tracker
#import wandb
# torch 
import torch 
from torch import nn
# linalg
import numpy as np
# colors
from termcolor import colored



# training one epoch
def train_one_epoch(loader, model, optimizer, scheduler, criteria,device,num_batches,verbose=True):
    """
  
  
  
    """
    # we train model 
    model = model.train()
    # keep track of the losses 
    # we add wandinit here
    running_loss = []
    correct_preds = 0
    # iterate over the loader
    for idx,instance in enumerate(loader):
      # get the inputs, attention mask and the labels) # i think i can do model(**instances)
      inputs = instance['input_ids'].to(device)
      attention_mask = instance['attention_mask'].to(device)
      labels = instance['label'].to(device)
      # now getting the outputs 
      out = model(x=inputs, attention=attention_mask)
      # predictions & loss
      _, y_preds = torch.max(out, dim=1)
      # torch.float32 torch.int64
      loss = criteria(out, labels)
      # summing to correct number
      correct_preds += torch.sum(y_preds == labels)
      # append the loss
      running_loss.append(loss.item())
      # propagate backwards 
      loss.backward()
      # clip the weights 
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
      # optimizer, scheduler step 
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad() # zero gradients
      # feedback
      if verbose:
        if idx % 100 ==0:
          # text for verbosity
          train_text = f"\n~BATCH STEP~ {idx+1}: ACCURACY: {correct_preds.double()/num_batches} LOSS: {np.mean(running_loss)}"
          # logging the step accuracies & loss to wandb
          #wandb.log({"step_acc":correct_preds/num_batches})
          #wandb.log({"step_loss":np.mean(running_loss)})
          print(colored(train_text,'cyan'))
    # return the overall accuracy       
    acc = correct_preds.double() / num_batches
    return (acc, np.mean(running_loss))