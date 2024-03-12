import torch, os, seaborn
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from dataset.augmentation import data_augmenter
from utils.file_dealer import img_loader
from utils.tb_lib import translate_pred, get_confusion

__all__ = ["DogCatCls"]

class DogCatCls:
    def __init__(self, data_root, output_save, section, data_format, image_size, maxv, data_loader, model):

        self.output_save = output_save

        if section == "final_test":
            csv_pth = f"{data_root}/sampleSubmission.csv"
            self.comp_csv = pd.read_csv(csv_pth)
            self._final_test(data_root, data_format, image_size, maxv, model)
            self.comp_csv.to_csv(csv_pth, index=False)
        elif section == "final_val":
            self._final_val(data_loader, model)        

    def _final_test(self, data_root, data_format, image_size, maxv, model):
        augmenter =  data_augmenter(data_format, "sham", 1., image_size)
        test_folder = f"{data_root}/test1"
        fnames = os.listdir(test_folder)
        for name in fnames:
            img = (augmenter(image=img_loader(f"{test_folder}/{name}", keep_color = True))['image']/maxv).unsqueeze(0).float().cuda()
            pred = model(img).cpu()
            prob = translate_pred(pred)[0].detach().numpy()
            pred_class = prob[0] if pred.shape[1] == 1 else prob[1]
            self._write_csv(name, pred_class)

    def _write_csv(self, fname, pred_class):
        idx = int(fname.split(".")[0])
        self.comp_csv.loc[self.comp_csv.id==idx,'label'] = pred_class

    def _final_val(self, data_loader, model):
        model.eval()
        with torch.no_grad():
            self.enum_preds = torch.tensor([])
            self.enum_targets = torch.tensor([])
            for i, (data, target) in enumerate(data_loader):
                input, target = data.cuda().float(), target
                pred = translate_pred(model(input)).cpu()
                self.enum_preds = torch.concatenate((self.enum_preds,pred))
                self.enum_targets = torch.concatenate((self.enum_targets,target))
            
            confusion_matrix = get_confusion(self.enum_preds.clone(), self.enum_targets.clone().long())
            self._plot_figure(confusion_matrix) 

    def _plot_figure(self, confusion_matrix):
        classes = ["cat", "dog"]
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