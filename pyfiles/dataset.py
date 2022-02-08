#!/usr/bin/env/python3

# ignore future warnings 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# torch & dataset
import torch 
from torch.utils.data import Dataset

# we create the dataset class 
class SentimentDataset(Dataset):

  # initialize - passing the text, labels, tokenizer and maximum length 
  def __init__(self, text, labels, tokenizer, max_length):

    self.text = text
    self.labels = labels
    self.tokenizer = tokenizer 
    self.max_length = max_length

  # length of the dataset 
  def __len__(self):
      return len(self.text)

  # now gettin the items 
  def __getitem__(self, idx):
      # get all the parts: text, input ids, attention mask, targets 
      text = str(self.text[idx])
      label = self.labels[idx]
      label = torch.tensor(label, dtype=torch.long) # it is a double, and the nll_backward loss does not accept those!!
      # encoder - encode plus is used because it uses the stop tokens & pads to the specified length
      encoder = self.tokenizer.encode_plus(text, 
                                  add_special_tokens=True, 
                                  max_length=self.max_length,
                                  pad_to_max_length = True,
                                  #padding = 'max_length',
                                  return_token_type_ids=False,
                                  return_attention_mask=True,
                                  return_tensors='pt')
      
      # a dictionary will be returned with all the items 
      input_dict = {
                      "text":text,
                      "input_ids":encoder['input_ids'].flatten(),
                      "attention_mask":encoder['attention_mask'].flatten(),
                      "label":label,

                      }
      
      return input_dict