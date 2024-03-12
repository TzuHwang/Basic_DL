import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn 
__all__ = ["MUTAG"]

class MUTAG:
    def __init__(self, 
                 writer, 
                 mod, 
                 epoch,
                 iter, 
                 lr=None, 
                 loss_matrix = None, 
                 confusion_matrix = None,
                 similarity = None,
                 gt_display = None,
                 pred_display = None):
        
        self.writer = writer
        self.mod = mod
        self.epoch = epoch
        self.iter = iter
        self.lr = lr
        self.loss_matrix = loss_matrix
        self.confusion_matrix = confusion_matrix
        self.similarity = similarity
        self.gt_display = gt_display
        self.pred_display = pred_display

        self._check_args()
        self._draw_scaler()
        self._draw_figure()
        
    def _check_args(self):
        if self.mod == 'train':
            assert self.lr != None
        assert self.loss_matrix != None
        assert self.confusion_matrix != None

    def _draw_scaler(self):
        if self.mod == 'train':
            self.writer.add_scalar("learning_rate", self.lr, self.iter)

        for loss in self.loss_matrix:
            self.writer.add_scalar(f"{self.mod}_{loss}", self.loss_matrix[loss], self.iter)

        for standard in self.confusion_matrix["scaler"]:
            self.writer.add_scalar(f"{self.mod}_{standard}", self.confusion_matrix["scaler"][standard], self.iter)

    def _draw_figure(self):
        classes = ["Non-Carcinogenic", "Carcinogenic"]
        cf_matrix, auc_curves = self.confusion_matrix["plot"]["confusion_matrix"], self.confusion_matrix["plot"]["auc_curves"]
        # plot auc_curves
        fig1, axes = plt.subplots(1, len(auc_curves), figsize=(5*len(auc_curves), 5))
        for i, (ax, [fpr, tpr, threshold]) in enumerate(zip(axes.flat, auc_curves)):
            ax.set_title(f"{classes[i]}")
            ax.plot(fpr, tpr); ax.plot([0, 1], ls="--")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
            if i == 0:
                ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
            else:
                ax.set(xlabel='False Positive Rate')
        self.writer.add_figure(f"{self.mod}_auc_curves", fig1, self.iter)
        # plot cf_matrix
        fig2 = plt.figure(figsize=(5*len(classes)+2, 5*len(classes)))
        seaborn.heatmap(pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes]), annot=True, vmin=0, vmax=1, cmap='plasma').get_figure()
        plt.xlabel("Prediction"); plt.ylabel("GroundTruth")
        self.writer.add_figure(f"{self.mod}_confusion_matrix", fig2, self.iter)

    def console_log(self):
        print(f"{self.mod}")
        print(f"epoch: {self.epoch}; iteration: {self.iter}; total_loss: {str(np.round(self.loss_matrix['loss'].detach().cpu().numpy(), 4))}")
        for i, name in enumerate(self.confusion_matrix["scaler"]):
            if i == 0:
                matrix_string = f"{name}: {str(np.round(self.confusion_matrix['scaler'][name], 4))}"
            else:
                matrix_string += f"; {name}: {str(np.round(self.confusion_matrix['scaler'][name], 4))}"
        print(matrix_string)