import os, sys
from . import hippocampus, fish, dogvscat, mutag

from torch.utils.tensorboard import SummaryWriter

class Result_Visualization:
    def __init__(self, args):
        self.log_dir = f"{args.output_root}/{args.config_name}/log"
        if os.path.exists(self.log_dir):
            if args.rm_exist_log and len(os.listdir(self.log_dir)) != 0:
                user_input = input("There are files in the directory, Do you want to continue? (yes/no): ")
                if user_input.lower() in ['yes', "y"]:
                    [os.remove(f'{self.log_dir}/{f}') for f in os.listdir(self.log_dir)]
                else:
                    raise Exception("Exit")
        else:
            os.makedirs(self.log_dir)
        self.writer_name = args.writer_name
        self.writer = SummaryWriter(self.log_dir)

    def write_log(self,
                  mod, 
                  epoch, 
                  iter, 
                  lr = None, 
                  loss_matrix = None,
                  confusion_matrix = None,
                  similarity = None,
                  gt_display = None,
                  pred_display = None):
    
        if self.writer_name in hippocampus.__all__: 
            log = hippocampus.__dict__[self.writer_name](
                self.writer,
                mod,
                epoch, 
                iter,
                lr = lr,
                loss_matrix = loss_matrix,
                similarity = similarity,
                gt_display = gt_display,
                pred_display = pred_display
            )
        elif self.writer_name in fish.__all__:
            log = fish.__dict__[self.writer_name](
                self.writer,
                mod,
                epoch, 
                iter,
                lr = lr,
                loss_matrix = loss_matrix,
                confusion_matrix = confusion_matrix,
                similarity = similarity,
                gt_display = gt_display,
                pred_display = pred_display
            )
        elif self.writer_name in dogvscat.__all__:
            log = dogvscat.__dict__[self.writer_name](
                self.writer,
                mod,
                epoch, 
                iter,
                lr = lr,
                loss_matrix = loss_matrix,
                confusion_matrix = confusion_matrix,
                similarity = similarity,
                gt_display = gt_display,
                pred_display = pred_display
            )
        elif self.writer_name in mutag.__all__:
            log = mutag.__dict__[self.writer_name](
                self.writer,
                mod,
                epoch, 
                iter,
                lr = lr,
                loss_matrix = loss_matrix,
                confusion_matrix = confusion_matrix,
                similarity = similarity,
            )            

        log.console_log()