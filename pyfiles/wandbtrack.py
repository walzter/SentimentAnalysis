#!/usr/bin/env/python3

# tracker
import wandb

# config file 
# Instantiate Wandb with some notes 
def init_wandb(config):
  
  notes = f"""

          Data: No clean
          EPOCHS: {config['EPOCHS']}
          LR: {config['LEARNING_RATE']}
          RUN: {config['RUN']}

          """

  wandb_init_dict = {
      
                      "project" : "IOMED-NLP-TEST",
                      "entity"  : "walzter",
                      "notes"   : notes,
                      "name"    : f"BERT_{config['EPOCHS']}_{config['LEARNING_RATE']}_NoClean_SampleData_{config['RUN']}",

                    }

  wandb.init(**wandb_init_dict)

# log custom plots with wandb!
def log_plots_wandb(y_actual, y_pred, y_probas, class_names):

  # wandb logging
  # CONFUSION MATRIX 
  wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None,
                                                    y_true=y_actual,
                                                    preds=y_pred,
                                                    class_names=class_names)
            })

  # ROC CURVE
  wandb.log({'roc': wandb.plot.roc_curve(
                                          y_actual,
                                          y_probas,
                                          labels=class_names)
            })

  # PR CURVE
  wandb.log({'pr': wandb.plot.pr_curve(
                                      y_actual,
                                      y_probas,
                                      labels=class_names)
            })
  print("\nCustom plots have been saved: Confusion Matrix, ROC, & PR Curves ")