import numpy as np
__all__ = ["FishSeg", "FishCls"]

class FishSeg:
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
        self.similarity = similarity
        self.gt_display = gt_display
        self.pred_display = pred_display

        self._check_args()
        self._draw_scaler()
        self._draw_img_grid()

    def _check_args(self):
        if self.mod == 'train':
            assert self.lr != None
        elif self.mod == 'eval':
            assert self.similarity != None
        assert self.loss_matrix != None

    def _draw_scaler(self):
        if self.mod == 'train':
            self.writer.add_scalar("learning_rate", self.lr, self.iter)
        elif self.mod == 'eval':
            for name in self.similarity:
                for i in range(len(self.similarity[name])):
                    self.writer.add_scalar(f"{self.mod}_{name}_channel_{i}", self.similarity[name][i], self.iter)

        for loss in self.loss_matrix:
            self.writer.add_scalar(f"{self.mod}_{loss}", self.loss_matrix[loss], self.iter)
    
    def _draw_img_grid(self):
        # TensorBoard's default data format is CHW. The default setting works with tensors. If your input is a NumPy array, set it to HWC
        self.writer.add_image(f"{self.mod}_gt", self.gt_display, self.iter, dataformats='HWC')
        self.writer.add_image(f"{self.mod}_pred", self.pred_display, self.iter, dataformats='HWC')

    def console_log(self):
        print(f"{self.mod}")
        print(f"epoch: {self.epoch}; iteration: {self.iter}; total_loss: {str(np.round(self.loss_matrix['loss'].detach().cpu().numpy(), 4))}")
        for name in self.similarity:
            for i in range(len(self.similarity[name])):
                if i == 0:
                    dice_string = f"{name}_channel_{i}: {str(np.round(self.similarity[name][i].detach().cpu().numpy(), 4))}"
                else:
                    dice_string += f"; {name}_channel_{i}: {str(np.round(self.similarity[name][i].detach().cpu().numpy(), 4))}"
            print(dice_string)


class FishCls:
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
        self._draw_img_grid()

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

        for standard in self.confusion_matrix:
            self.writer.add_scalar(f"{self.mod}_{standard}", self.confusion_matrix[standard], self.iter)

    def _draw_img_grid(self):
        # TensorBoard's default data format is CHW. The default setting works with tensors. If your input is a NumPy array, set it to HWC
        self.writer.add_image(f"{self.mod}_gt", self.gt_display, self.iter, dataformats='HWC')
        self.writer.add_image(f"{self.mod}_pred", self.pred_display, self.iter, dataformats='HWC')

    def console_log(self):
        print(f"{self.mod}")
        print(f"epoch: {self.epoch}; iteration: {self.iter}; total_loss: {str(np.round(self.loss_matrix['loss'].detach().cpu().numpy(), 4))}")
        for i, name in enumerate(self.confusion_matrix):
            if i == 0:
                matrix_string = f"{name}: {str(np.round(self.confusion_matrix[name], 4))}"
            else:
                matrix_string += f"; {name}: {str(np.round(self.confusion_matrix[name], 4))}"
        print(matrix_string)