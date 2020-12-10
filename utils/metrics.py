import torch
import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict

class DefaultClassificationMetrics:
    
    def __init__(self, n_classes, int_to_labels=None, act_threshold=0.5, mode=""):
        self.mode = mode
        self.n_classes = n_classes
        self.prefix = ""
        if mode:
            self.prefix = mode + "_"
        self.n_classes = n_classes
        if int_to_labels is None:
            int_to_labels = {val:'class_'+str(val) for val in range(n_classes)}
        self.int_to_labels = int_to_labels
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.truths = []
        self.predictions = []        
        
    
    # add predictions to confusion matrix etc
    def add_preds(self, y_pred, y_true):
        y_pred = y_pred.max(dim = 1)[1].data
        y_true = y_true.flatten().detach().cpu().numpy()
        y_pred = y_pred.flatten().detach().cpu().numpy()
        self.truths += (y_true.tolist())
        self.predictions += (y_pred.tolist())
        np.add.at(self.confusion_matrix, (y_true, y_pred), 1)
    
    
    # Calculate and report metrics
    def get_value(self):
        # I only added accuracy here since we use CIFAR10
        accuracy = metrics.accuracy_score(self.truths, self.predictions)
        
        # return metrics as dictionary
        return edict({
            self.prefix + "accuracy" : round(accuracy, 3),
                     })