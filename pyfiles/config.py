#!/usr/bin/env/python3

# to get the device
import torch


def make_config():
    """
    
    
    """
    
    config_dict = {
                    "DATA"             :'./dataset.zip',
                    "DATA_DIR"         :'./data',
                    "EPOCHS"           :3,
                    "COLS_TO_USE"      :[0,5],
                    "LABEL_NAMES"      :['label','text'],
                    "BATCH_SIZE"       :16,
                    "MAX_LENGTH"       :80,
                    "OLD_MAX_LENGTH"   :280,
                    "DEVICE"           :torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    "NUM_CLASSES"      :2,
                    "PRE_TRAINED_MODEL":'bert-base-cased',
                    "DROPOUT"          :0.3,
                    "LEARNING_RATE"    :2e-5,
                    "VAL_SIZE"         :0.2,
                    "RANDOM_STATE"     :42,
                    "VERBOSE_STEP"     :True,
                    "SAVE_MODEL"       :True,
                }
    
    return config_dict