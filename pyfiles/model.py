#!/usr/bin/env/python3

# config dictionary
from config import make_config
# torch 
from torch import nn
# import the transformers 
from transformers import BertModel
from transformers import logging 
logging.set_verbosity_error()
# config file
config_dict = make_config()

class BertFeels(nn.Module):
    def __init__(self, num_classes):
        super(BertFeels, self).__init__()
        self.bert_model = BertModel.from_pretrained(config_dict['PRE_TRAINED_MODEL'])
        self.dropout = nn.Dropout(p=config_dict['DROPOUT'])
        self.output = nn.Linear(self.bert_model.config.hidden_size, num_classes)
    
    # forward pass
    def forward(self,x, attention):
        # get the output of the model ()
        output = self.bert_model(
                                 input_ids = x,
                                 attention_mask = attention)
        pool_out = output['pooler_output']
        pool_out = pool_out.to(config_dict['DEVICE'])
        # pass it through the dropout
        out = self.dropout(pool_out)
        # return the linear classification 
        return self.output(out)