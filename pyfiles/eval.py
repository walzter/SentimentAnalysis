#!/usr/bin/env/python3


# torch 
import torch 
# linalg
import numpy as np
# EVAL one epoch

def eval_one_epoch(loader, model, criteria, device, num_batches):
    """
    """
    # we train model 
    model = model.train()
    # keep track of the losses 
    running_loss = []
    correct_preds = 0
    # iterate over the loader
    with torch.no_grad():
      for instance in loader:
        # get the inputs, attention mask and the labels)
        inputs = instance['input_ids'].to(device)
        attention_mask = instance['attention_mask'].to(device)
        labels = instance['label'].to(device)
        # now getting the outputs 
        out = model(x=inputs, attention=attention_mask)
        # predictions & loss
        _, y_preds = torch.max(out, dim=1)
        loss = criteria(out, labels)
        # summing to correct number
        correct_preds += torch.sum(y_preds == labels)
        # append the loss
        running_loss.append(loss.item())
    # accuracy
    acc = correct_preds.double() / num_batches
    return (acc, np.mean(running_loss))