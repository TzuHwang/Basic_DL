import argparse
from typing import Any

class Access_Args:
    """
    arg: argument
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._args_pths()
        self._args_section_ctrl()
        self._args_dataset()
        self._args_model()
        self._args_loss()
        self._args_tensorboard()
        self._args_save()

    def get_args(self):
        return self.parser.parse_args()

    def _args_pths(self):
        self.parser.add_argument(
            "--data-root", type=str, help="Dataset directory"
            )
        self.parser.add_argument(
            "--output-root", type=str, help="Model saving directory",
            default='./output'
            )
        self.parser.add_argument(
            "--config-name", type=str, help="The name of the config"
            )           

    def _args_section_ctrl(self):
        self.parser.add_argument(
            "--section", type=str, help="Selecting the training or evaluation on proceeding"
            )
        self.parser.add_argument(
            "--max_iter", type=int, help="Setting the max iter number per epoch",
            default=None
            )                

    def _args_dataset(self):
        self.parser.add_argument(
            "--dataset", type=str, help="The name of dataset"
            )
        self.parser.add_argument(
            "--data-format", type=str, help="The format of data, ex: img, table"
            )        
        self.parser.add_argument(
            "--aug", type=str, help="The data augmentation method"
            )
        self.parser.add_argument(
            "--crop", type=float, help="The crop size",
            default=1.
            )        
        self.parser.add_argument(
            "--loading-method", type=str, help="The method of data loading, ex: img_2D, img_2.5D, table, ......etc",
            default='img_2D'
            )
        self.parser.add_argument(
            "--normalize", action='store_true', help="Normalizing the data before aug or not",
            )
        self.parser.add_argument(
            "--maxv", type=float, help="Normalizing the data by given value",
            default=None
            )        
        self.parser.add_argument(
            "--data-balancing", type=str, help="The data balance method, ex: smote, tomek, ......etc",
            default=None
            )
        self.parser.add_argument(
            "--batch-size", type=int, help="The number of sample in each batch.",
            default=16
            )        
        self.parser.add_argument(
            "--num-workers", type=int, help="The thread number of dataloader, base on gpu, not cpu",
            default=1
            )
        self.parser.add_argument(
            '--no-label-data-portion', type=float, help='Remove 1-x percent of data with no label',
            default=1. 
            )
        
    def _args_model(self):
        # model body
        self.parser.add_argument(
            "--task", type=str, help="The task of the experiment, ex: 'classification', 'segmentation', ......etc"
            )
        self.parser.add_argument(
            "--model-name", type=str, help="The name of selected model"
            )
        self.parser.add_argument(
            "--epoch", type=int, help="The number of epoches",
            default=100
            )
        self.parser.add_argument(
            '--conv-bn', action='store_true', help='Add batch norm in between conv and relu', 
            )
        self.parser.add_argument(
            '--bilinear', action='store_true', help='Use bilinear upsampling', 
            )
        self.parser.add_argument(
            "--output-channel-num", type=int, help="The chinnel number of model outputs"
            )
        self.parser.add_argument(
            '--use-pretrained', action='store_true', help='Use pretrained model', 
            )
        self.parser.add_argument(
            "--dropout", type=float, help='During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.',
            default=0
        )       
        
        # optimizer
        self.parser.add_argument(
            "--optimizer-name", type=str, help="The name of optimizer"
            )
        self.parser.add_argument(
            "--learning-rate", type=float, help="A higher learning rate will decrease the probability of overfitting but may also hinder the model from finding the local minimum",
            default=0.001
            )
        self.parser.add_argument(
            "--momentum", type=float, help="A higher momentum will decrease the probability of overfitting but may make the model unstable",
            default=0.9
            )
        self.parser.add_argument(
            "--weight-decay", type=float, help="Weight decay (L2 penalty)",
            default=0. # https://zhuanlan.zhihu.com/p/63982470
            ) 

        # scheduler
        self.parser.add_argument(
            "--scheduler-name", type=str, help="The name of scheduler"
            )
        self.parser.add_argument(
            "--warmup-epoch", type=int, help="The number of warmup epoches",
            default=50
            )
        self.parser.add_argument(
            "--step-size", type=int, help="For StepLR scheduler, the learning-rate decrease per x epoch num",
            default=10
            )        
        self.parser.add_argument(
            "--learning-rate-decay", type=float, help="For StepLR scheduler, The degree of learning-rate decrease per step size",
            default=0.1
            )
        self.parser.add_argument(
            "--multiplier", type=float, help="target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.",
            default=1.
            )

    def _args_loss(self):
        self.parser.add_argument(
            "--losses", type=str, nargs='+', help="The list of applied losses",
            default=['SoftDiceLoss']
            )
        self.parser.add_argument(
            "--channel-weights", type=float, nargs='+', help="The list saving the weight of each channel",
            default=[1., 1., 1.]
            )        
        self.parser.add_argument(
            "--loss-weights", type=float, nargs='+', help="The list saving the weights of applied losses in order",
            default=[1.]
            )
    
    def _args_tensorboard(self):
        self.parser.add_argument(
            "--writer-name", type=str, help="The name of applied tensorboard writter"
            )
        self.parser.add_argument(
            "--similarity-fucn", type=str, nargs='+', help="The fucn used to calculate pred/ gt similarity"
            ) 
        self.parser.add_argument(
            "--print-frequency", type=int, help="The output was printed per x iter num",
            default=100
            )
        self.parser.add_argument(
            '--rm-exist-log', action='store_true', help='Remove the exist log in output root', 
            )

    def _args_save(self):
        self.parser.add_argument(
            "--save-frequency", type=int, help="A checkpoint was saved per x epoch num",
            default=50
            )
        self.parser.add_argument(
            '--save-best', action='store_true', help='Save the state with lowest loss', 
            )
        self.parser.add_argument(
            '--init-point', type=float, help='Save the state with lowest loss',
            default=0.1
            )