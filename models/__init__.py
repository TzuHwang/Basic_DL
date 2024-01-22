import os, sys, natsort
from .segmentation import Seg_models
from .classification import Cls_models

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class Construct_Model:
    def __init__(self, args, input_channel_num, iter_num_per_epoch):
        # Model
        # Deep learning models
        self.task = args.task
        self.model_name = args.model_name
        self.epoch_num = args.epoch
        self.input_channel_num = input_channel_num
        self.output_channel_num = args.output_channel_num
        self.bilinear = args.bilinear
        self.conv_bn = args.conv_bn
        self.dropout = args.dropout

        self.use_pretrained = args.use_pretrained 

        if self.task == "segmentation":
            self.model = Seg_models(
                self.model_name,
                self.input_channel_num,
                self.output_channel_num,
                self.bilinear,
                self.conv_bn
                ).get_model()
        elif self.task == "classification":
            self.model = Cls_models(
                self.model_name,
                self.input_channel_num,
                self.output_channel_num,
                self.use_pretrained,
                self.dropout
                ).get_model()        

        # Optimizer
        # Optimizers adjust the parameters of the model during training in order to minimize the error between the predicted output and the actual output.
        self.optimizer_name = args.optimizer_name
        self.lr = args.learning_rate # share variable with scheduler
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.multiplier = args.multiplier

        self.optimizer = self._get_optimizer()

        # Scheduler
        # Schedulers adjust the learning rate based on the number of epochs.
        self.scheduler_name = args.scheduler_name
        self.warmup_epoch = args.warmup_epoch
        self.iter_num_per_epoch = iter_num_per_epoch
        self.lr_decay = args.learning_rate_decay

        self.max_iter = (args.epoch-args.warmup_epoch)*iter_num_per_epoch
        self.step_size = args.step_size*iter_num_per_epoch
        self.warmup_iters = args.warmup_epoch*iter_num_per_epoch

        self.scheduler = self._get_scheduler(self.optimizer)

        # Resume training
        self.model_save_dir = f"{args.output_root}/{args.config_name}"
        self.init_epoch = 0
        
    def get_model(self):
        # self._resume()
        return self.model, self.optimizer, self.scheduler
    
    def get_init_epoch(self):
        return self.init_epoch
    
    def _get_optimizer(self):
        if self.optimizer_name == 'sgd':
            optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        return optimizer
    
    def _get_scheduler(self, optimizer):
        '''
        For the CosineAnnealingLR option, this scheduler is based on iteration. 
        Therefore, the others must be transformed to an iteration-based format.
        '''
        if self.scheduler_name == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_iter,
                eta_min=1e-10
            )
        elif self.scheduler_name == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.lr_decay, # constant of learning decay
            )
        if self.warmup_epoch>0:
            scheduler = GradualWarmupScheduler(
                optimizer,
                self.multiplier,
                total_iter = self.warmup_iters,
                after_scheduler = scheduler
            )
        return scheduler
    
    def _resume(self):
        ckpts = natsort.natsorted(os.listdir(self.model_save_dir))
        if "log" in ckpts:
            ckpts.remove("log")
        if len(ckpts) > 0:
            ckpt = torch.load(f"{self.model_save_dir}/{ckpts[-1]}")
            self.init_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            # Now optimizer.state is located in cpu, but this code base is based on gpu computing. 
            # This issue will trigger RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()    

class GradualWarmupScheduler(_LRScheduler):
    # https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

    rm_Args:
        total_epoch: target learning rate is reached at total_epoch, gradually

    Altered_Args:
        last_epoch = Originally, it represented the last epoch of training, but after the alteration, it now represents the last iteration.
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler = None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch ) / self.total_iter) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch  / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, iter=None):
        if iter is None:
            iter = self.last_epoch  + 1
        self.last_epoch  = iter if iter != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch  <= self.total_iter:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if iter is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, iter - self.total_iter)

    def step(self, iter=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if iter is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(iter - self.total_iter)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(iter)
        else:
            self.step_ReduceLROnPlateau(metrics, iter)