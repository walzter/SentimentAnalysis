#!/usr/bin/env/python3

# plotting
import matplotlib.pyplot as plt
import seaborn as sns 
# more metrics
from sklearn.metrics import roc_curve,confusion_matrix,PrecisionRecallDisplay,classification_report
# dataframe 
import pandas as pd


# Plot Confusion Matrix 
def make_plot_confusion_matrix(y_test, y_pred,class_names=['Negative','Positive']):
  # make a confusion matrix 
  cm = confusion_matrix(y_test, y_pred)
  # make a df out of it 
  cm = pd.DataFrame(cm, index=class_names, columns=class_names)
  # make a heatmap
  g = sns.heatmap(cm, annot=True,fmt='d')
  # correcting the ticks
  g.yaxis.set_ticklabels(g.yaxis.get_ticklabels(), rotation=0, ha='right')
  g.xaxis.set_ticklabels(g.xaxis.get_ticklabels(), rotation=0, ha='right')
  # setting the labels 
  plt.ylabel('Actual Sentiment')
  plt.xlabel('Predicted Sentiment')
  plt.title("Confusion Matrix: Actual vs. Predicted")
  plt.savefig("Confusion_Matrix_BERT_3_NoClean.jpg")
  plt.show()
  

# plot ROC
# This function is adapted from the SKLEARN-ROC tutorial
def plot_roc(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=f'ROC curve (area = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Graph for the ROC Curve ')
    plt.legend(loc="lower right")
    plt.savefig("ROC_CURVE_BERT.jpg")
    plt.show()

# plot the PR curve 
def plot_pr_curve(y_true, y_preds):
  """

  """
  # PR Curve
  display = PrecisionRecallDisplay.from_predictions(y_true, y_preds, name="Bert")
  _ = display.ax_.set_title("Graph for the PR Curve")
  plt.savefig('PR_CURVE_BERT.jpg')
  plt.show()
  
def display_metrics(history, y_true, y_preds,class_names):
  """

  """
  # message 
  print("\nThe Metrics Reported: ")
  print("\n-->Training Accuracy & Loss: \n")
  # Training Accuracy & Loss
  plt.plot(history['train_acc'],  label = 'Train Accuracy')
  plt.plot(history['train_loss'], label = 'Train Loss')
  plt.xlabel('Steps')
  plt.ylabel('Value')
  plt.title("Graph for the Training Accuracy & Loss")
  plt.show()
  print("\n-->Validation Accuracy & Loss: \n")
  # Validation Loss & Accuracy 
  plt.plot(history['val_acc'],  label = 'Validation Accuracy')
  plt.plot(history['val_loss'], label = 'Validation Loss')
  plt.xlabel('Steps')
  plt.ylabel('Value')
  plt.title("Graph for the Validation Accuracy & Loss")
  plt.show()
  # make the classification report 
  cr = classification_report(y_true, y_preds, target_names = class_names)

  # print the classification report 
  print(cr)

  # confusion matrix 
  print("\n--> Confusion Matrix: \n")
  make_plot_confusion_matrix(y_true, y_preds,class_names)


  # ROC Curve
  print("\n--> ROC Curve: \n")
  # first get the respective rates & values
  fpr, tpr, roc_auc = roc_curve(y_true, y_preds, pos_label=1)
  # pass them into the plotting function 
  plot_roc(fpr, tpr, roc_auc) 

  # PR Curve
  print("\n--> PR Curve: \n")
  plot_pr_curve(y_true, y_preds)