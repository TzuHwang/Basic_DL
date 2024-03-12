import torch, os, seaborn, copy
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from dataset.augmentation import data_augmenter
from utils.file_dealer import img_loader
from utils.misc import to_model
from utils.tb_lib import translate_pred, get_confusion, one_hot_encoding
from utils.pyg_support import Explainer, GNNExplainer, Draw_graph

__all__ = ["MUTAG"]

label_representation = {0: 'C',
                        1: 'N',
                        2: 'O',
                        3: 'F',
                        4: 'I',
                        5: 'Cl',
                        6: 'Br'}

class MUTAG:
    def __init__(self, data_root, output_save, section, data_format, data_loader, model):

        self.output_save = output_save

        if section == "final_test":
            pass
        elif section == "final_val":
            self.explainer = self._set_explainer(data_loader, data_format, copy.deepcopy(model)) 
            self._final_val(data_loader, data_format, model)      

    def _final_test(self, data_root, data_format, model):
        pass

    def _write_csv(self, fname, pred_class):
        pass

    def _final_val(self, data_loader, data_format, model):
        model.eval()
        self.enum_preds = torch.tensor([]).cuda()
        self.enum_targets = torch.tensor([]).cuda()
        for i, (data, target) in enumerate(data_loader):
            input, target = to_model(data, data_format), target.cuda()
            with torch.no_grad():
                pred = translate_pred(model(input))
                self.enum_preds = torch.concatenate((self.enum_preds,pred))
                self.enum_targets = torch.concatenate((self.enum_targets,target))
            explanation = self.explainer(input.x, input.edge_index, batch=input.batch)
            Draw_graph(input.cpu(), target.cpu(), explanation.cpu(), label_representation)
                
        confusion_matrix = get_confusion(self.enum_preds.detach().cpu().clone(), self.enum_targets.detach().cpu().clone().long())
        self._plot_figure(confusion_matrix) 
        

    def _set_explainer(self, data_loader, data_format, model):

        explainer = Explainer(
                    model= model,
                    algorithm=GNNExplainer(epochs=200).cuda(),
                    explanation_type='model',
                    edge_mask_type='object',
                    model_config=dict(
                        mode='multiclass_classification',
                        task_level='graph',
                        return_type='raw',
                    ),
                    # Include only the top 10 most important edges:
                    threshold_config=dict(threshold_type='topk', value=10),
                )

        return explainer

    def _plot_figure(self, confusion_matrix):
        classes = ["Non-Carcinogenic", "Carcinogenic"]
        cf_matrix, auc_curves = confusion_matrix["plot"]["confusion_matrix"], confusion_matrix["plot"]["auc_curves"]
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
        fig1.savefig(f"{self.output_save}/auc_curves.png")
        # plot cf_matrix
        fig2 = plt.figure(figsize=(5*len(classes)+2, 5*len(classes)))
        seaborn.heatmap(pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes]), annot=True, vmin=0, vmax=1, cmap='plasma').get_figure()
        plt.xlabel("Prediction"); plt.ylabel("GroundTruth")
        fig2.savefig(f"{self.output_save}/cf_matrix.png")
        plt.close("all")

        # merge two plot
        fig, axes = plt.subplots(2, 1, figsize=(8,8))
        axes[0].imshow(plt.imread(f"{self.output_save}/auc_curves.png"))
        axes[1].imshow(plt.imread(f"{self.output_save}/cf_matrix.png"))
        [ax.axis('off') for ax in axes.flat]
        fig.tight_layout()
        increment_y = 0.05
        for i, key in enumerate(confusion_matrix["scaler"]):
            fig.text(0.05, 0.6-i*increment_y, f"{key}: {np.round(confusion_matrix['scaler'][key], 3)}", fontsize=8)
        plt.subplots_adjust(left=0.2)
        plt.show()