import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

__all__ = ["CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss", "CrossentropyND"]

'''
The shape of pred and target should be (c), (b, c), (b, c, d)
b = batch size
c = num of class
d = 1 dim prob
'''

'''
You should not pass the softmax into the CrossEntropy loss. It computes log_softmax(y2) internally, so you end up with with log_softmax(softmax(z))
'''

class CrossentropyND(CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        if inp.shape != target.shape:
            target = target.long()
            class_base = True
        else:
            class_base = False
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            if class_base is False:
                target = target.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous().view(-1, num_classes)

        target = target.contiguous().view(-1, num_classes) if class_base is False else target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)
    
if __name__ == "__main__":
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()