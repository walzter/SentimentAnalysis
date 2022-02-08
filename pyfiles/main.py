#!/usr/bin/env/python3

# config 
from config import make_config
from pipeline import model_pipeline
from trainer import train_model

# predict
from predict import make_preds
# utils 
from utils import display_metrics
# track
from wandbtrack import log_plots_wandb


# the final main
def main():
    config_dict = make_config()
    
    # visualize our config dictionary
    for k,v in config_dict.items():
        print('\n',k,v)
    # pipeline
    model, optimizer, scheduler, criterion, train_loader, val_loader, test_loader,df_train,df_val = model_pipeline(config_dict)
    #
    history = train_model(model,
                      train_loader,
                      val_loader,
                      optimizer, 
                      scheduler, 
                      criterion, 
                      config_dict['EPOCHS'],
                      config_dict['DEVICE'],
                      df_train,
                      df_val,
                      config=config_dict,
                      save_model=config_dict['SAVE_MODEL'])

    # Evaluate on the test-set 
    # we get the predictions, real values & probabilities 
    y_preds, y_true, probas = make_preds(model, test_loader,criterion)
    # adjusting a bit 
    y_preds = y_preds.numpy().tolist()
    y_true = y_true.numpy().tolist()
    # refer to the class names 
    class_names = ['Negative','Positive']
    #log the custom plots 
    log_plots_wandb(y_true, y_preds, probas, class_names)
    # show the metrics & plots 
    display_metrics(y_true, y_preds,class_names)
    
    return history

if __name__ == "__main__":
    main()
    