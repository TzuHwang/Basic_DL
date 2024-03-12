import torch
import torch.nn.functional as F

from . import ce_loss, seg_loss

def translate_pred(pred):
    c = pred.shape[1]
    translated = torch.sigmoid(pred) if c == 1 else torch.softmax(pred, dim = 1)
    return translated

def one_hot_encoding(input, num_classes):
    return F.one_hot(input, num_classes = num_classes)

class Loss_Fcns:
    def __init__(self, args):
        self.loss_fcns = {}
        self.loss_values = {}
        self.in_channel_loss_values = {}
        self.losses = args.losses
        self.channel_weights = args.channel_weights
        self.weights = args.loss_weights

        for loss in self.losses:
            if loss in seg_loss.__all__:
                self.loss_fcns[loss] = seg_loss.__dict__[loss](smooth=1., channel_weights=self.channel_weights)
            elif loss in ce_loss.__all__:
                self.loss_fcns[loss] = ce_loss.__dict__[loss]()
            else:
                raise Exception("Unvalidate input")

    def get_loss_fcns(self):
        return self.loss_fcns
    
    def get_loss_value(self, pred, target):
        total_loss, logit, prob = 0, pred, translate_pred(pred)
        if target.shape != pred.shape:
            target = one_hot_encoding(target, pred.shape[1]) if pred.shape[1]>1 else target.unsqueeze(1)
        assert len(self.weights) == len(self.losses)
        for weight, loss in zip(self.weights, self.losses):
            if loss in seg_loss.__all__:
                self.loss_values[loss] = self.loss_fcns[loss](prob, target.long())
            elif loss in ce_loss.__all__:
                if loss == "BCELoss":
                    self.loss_values[loss] = self.loss_fcns[loss](prob, target.float())
                else:
                    self.loss_values[loss] = self.loss_fcns[loss](logit, target.float())
            total_loss += self.loss_values[loss]*weight
        self.loss_values['loss'] = total_loss
        return self.loss_values