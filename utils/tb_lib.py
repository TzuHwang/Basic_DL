import numpy as np
import torch, cv2
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

from .cv_lib import imgmerge

colors = [
    (255, 0, 0), # red
    (255, 0, 0), # red
    (0, 255, 0), # green
    (0, 0, 255), # blue
    (255, 0, 255), # purple
    (255, 255, 0), # yellow
    (0, 255, 255), # cyan
    (139, 69, 19), # saddle brown
    (255, 192, 203), # pink
    (127, 255, 212), # aqua marine
    (245, 222, 179), # wheat
    (255, 165, 0), # orange
]

def concat_imgs(bases, covers, num_classes, row_num = 2, for_pred = False, task = "segmentation"):
    b, c, h, w = bases.shape
    col_num = np.ceil(b / row_num)
    batch_imgs = np.zeros((int(h*row_num), int(w*col_num), len(colors[1])), dtype = np.uint8)
    if for_pred:
        covers = translate_pred(covers, to_one_hot = True)
        if task == "classification":
            covers = np.argmax(covers, 1)
    bases = torch.moveaxis(bases, 1, -1) if c == 3 else bases.squeeze()
    for i, (base, cover) in enumerate(zip(bases, covers)):
        base = (base.numpy()*255).astype(np.uint8)
        cover = (cover.numpy()*255).astype(np.uint8) if task == "segmentation" else cover.numpy()
        img = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if c == 1 else base.copy()
        # if len(colors["channel_1"]) == 4:
        #     img = cv2.cvtColor(base, cv2.COLOR_GRAY2RGBA if c == 1 else cv2.COLOR_RGB2RGBA)
        #     for k, channel in enumerate(cover):
        #         if k != 0:
        #             img = imgmerge(img, channel, cover=colors[f'channel_{k}'])
        if task == "segmentation":
            assert len(colors) >= covers.shape[1]
            for k, channel in enumerate(cover):
                if covers.shape[1] != 1 and k == 0:
                    continue
                cnts, _ = cv2.findContours(channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img = cv2.drawContours(img, cnts, -1, colors[k], 1)
        elif task == "classification":
            img = cv2.putText(img, str(cover), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        idx = int(np.floor(i/2))
        if i % 2 == 0:
            batch_imgs[0: h, idx*w: (idx+1)*w, :] = img
        else:
            batch_imgs[h: 2*h, idx*w: (idx+1)*w, :] = img
    return batch_imgs

def one_hot_encoding(input, num_classes):
    return F.one_hot(input, num_classes = num_classes)

def translate_pred(pred, to_one_hot = False):
    c = pred.shape[1]
    translated = torch.sigmoid(pred) if c == 1 else torch.softmax(pred, dim = 1)
    if to_one_hot:
        if c != 1:
            max_idx = torch.argmax(translated, 1)
            translated = torch.moveaxis(one_hot_encoding(max_idx, c), -1, 1)
        else:
            translated[translated>=0.5] = 1
            translated[translated<0.5] = 0
    return translated

def get_similarity(pred, target, smooth = 1., fucn = ["SoftDice"]):
    '''dice = (2 * intersection(pred, target)) / (pred + target)'''
    if target.shape != pred.shape:
        target =  one_hot_encoding(target, pred.shape[1])
    assert pred.shape == target.shape
    b, c, h , w = pred.shape
    pred = translate_pred(pred)
    similarity_matrix = {}
    if "SoftDice" in fucn:
        dice_in_channel = []
        for i in range(c):    
            intersection = (pred[:, i, :, :] * target[:, i, :, :]).sum()
            dice = (2 * intersection + smooth) / (pred[:, i, :, :].sum() + target[:, i, :, :].sum() + smooth)
            dice_in_channel.append(dice)
        similarity_matrix["SoftDice"] = dice_in_channel
    if "IoU" in fucn:
        pass
    return similarity_matrix

def get_confusion(pred, target):
    prob = translate_pred(pred)
    pred, num_classes = torch.argmax(prob, 1), prob.shape[1]

    assert pred.shape == target.shape

    acc = accuracy_score(target, pred)
    pre = precision_score(target, pred, average = "macro", zero_division=0)
    rec = recall_score(target, pred, average = "macro", zero_division=0)
    one_hot = one_hot_encoding(target, num_classes)
    auroc = []
    for i in range(num_classes):
        if len(np.unique(one_hot[:, i])) == 1:
            continue
        auroc.append(roc_auc_score(one_hot[:, i], prob[:, i], average = "macro", multi_class="ovr"))
    return {
            "accuracy": acc, 
            "precision":pre, 
            "recall":rec, 
            "auroc": np.mean(auroc)
            }