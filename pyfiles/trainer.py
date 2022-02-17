#!/usr/bin/env/python3

# dict, colored print statements, tracker & torch 
from collections import defaultdict
from termcolor import colored
import torch
# initiate tracking 
#from wandbtrack import init_wandb
# train loop 
from train import train_one_epoch
# val loop 
from eval import eval_one_epoch


# make a history tracker
def train_model(model,train_loader,val_loader, optimizer, scheduler, criteria, epochs, device, df_train, df_val, verbose, save_model):
  # init tracking
  #_ = init_wandb(config)
  # observe the model 
  #wandb.watch(model, criterion,log='all')
  # tracker
  history = defaultdict(list)
  # best accuracy tracker
  best_accuracy = 0
  #criteria = nn.BCEWithLogitsLoss().to(device)
  for epoch in range(epochs):  
    # training
    train_acc, train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, criteria, device, num_batches=len(df_train),verbose=verbose)
    # some feedback
    train_text = f"\nTRAINING -- EPOCH: {epoch+1} ACCURACY: {train_acc} LOSS: {train_loss}"
    print(colored(train_text, 'green','on_grey'))
    # Validation
    # testing
    val_acc, val_loss = eval_one_epoch(val_loader, model, criteria, device, num_batches = len(df_val))
    # feedback
    val_text = f"\nVALIDATION -- EPOCH: {epoch+1} ACCURACY: {val_acc} LOSS: {val_loss}"
    print(colored(val_text, 'white','on_blue',['bold']))
    
    # append the values 
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    # pass to wandb
    #wandb.log({"train_acc":train_acc})
    #wandb.log({"train_loss":train_loss})
    #wandb.log({"val_acc":val_acc})
    #wandb.log({"val_loss":val_loss})
    # keeping track of the better modeear
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        if save_model:
           torch.save(model.state_dict(), 'best_model_accuracy.bin')
  # finish the wandb run 
  #wandb.finish()
  return history
