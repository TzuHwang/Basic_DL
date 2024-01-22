import os, sys, numpy, natsort, cv2
import torch

from arguments import Access_Args
from dataset import Data_Loader
from models import Construct_Model
from loss import Loss_Fcns
from tb_visualization import Result_Visualization
from utils import file_dealer, tb_lib


def train(epoch, train_loader, model, optimizer, scheduler, loss, writer):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = data.cuda().float(), target.cuda()

        if args.task in ["segmentation", "classification"]:
            pred = model(input)
            loss_matrix = loss.get_loss_value(pred, target)
            loss_matrix['loss'].backward()
            optimizer.step()
            scheduler.step()

            # write log
            input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()
            iter_num = epoch * len(train_loader) + i
            lr = optimizer.param_groups[0]["lr"]
            gt_display, pred_display = None, None
            if args.data_format == "img":
                gt_display = tb_lib.concat_imgs(input.clone(), target.clone(), args.output_channel_num, task = args.task)
                pred_display = tb_lib.concat_imgs(input.clone(), pred.clone(), args.output_channel_num, for_pred=True, task = args.task)            
            if args.task == "segmentation":
                similarity = tb_lib.get_similarity(pred.clone(), target.clone(), fucn = args.similarity_fucn)
                writer.write_log('train', epoch, iter_num, lr = lr, loss_matrix = loss_matrix, 
                                similarity = similarity, gt_display = gt_display, pred_display = pred_display)
            elif args.task == "classification":
                confusion_matrix = tb_lib.get_confusion(pred.clone(), target.clone())
                writer.write_log('train', epoch, iter_num, lr = lr, loss_matrix = loss_matrix, 
                                confusion_matrix = confusion_matrix, gt_display = gt_display, pred_display = pred_display)
        elif args.task in ["multi_task"]:
            pass


def eval(epoch, eval_loader, model, loss, writer):
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(eval_loader):
            input, target = data.cuda().float(), target.cuda()
            epoch_loss = 0
            if args.task in ["segmentation", "classification"]:
                pred = model(input)
                loss_matrix = loss.get_loss_value(pred, target)
                epoch_loss += loss_matrix['loss']

                # write log
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()
                iter_num = epoch*len(eval_loader) + i
                gt_display, pred_display = None, None
                if args.data_format == "img":
                    gt_display = tb_lib.concat_imgs(input.clone(), target.clone(), args.output_channel_num, task = args.task)
                    pred_display = tb_lib.concat_imgs(input.clone(), pred.clone(), args.output_channel_num, for_pred=True, task = args.task)
                if args.task == "segmentation":
                    similarity = tb_lib.get_similarity(pred.clone(), target.clone(), fucn = args.similarity_fucn)
                    writer.write_log('eval', epoch, iter_num, lr = None, loss_matrix = loss_matrix, 
                                    similarity = similarity, gt_display = gt_display, pred_display = pred_display)
                elif args.task == "classification":
                    confusion_matrix = tb_lib.get_confusion(pred.clone(), target.clone())
                    writer.write_log('eval', epoch, iter_num, lr = None, loss_matrix = loss_matrix, 
                                    confusion_matrix = confusion_matrix, gt_display = gt_display, pred_display = pred_display)
            elif args.task in ["multi_task"]:
                pred = model(input)
                pass
        return epoch_loss/len(eval_loader)

def apply(eval_loader, model):
    pass

def main(args):
    if args.section == "training":
        train_loader = Data_Loader(args, 'train').get_loader()
        eval_loader = Data_Loader(args, 'val').get_loader()
    elif args.section == "test":
        train_loader = Data_Loader(args, 'train_final').get_loader()
        eval_loader = Data_Loader(args, 'test').get_loader()
    elif args.section == "application":
        eval_loader = Data_Loader(args, 'test').get_loader()
    print(f"Train data sizs:{len(train_loader)}")
    print(f"Eval data sizs:{len(eval_loader)}")

    tb_writer = Result_Visualization(args)
    input_channel_num = train_loader.dataset[0][0].shape[0]
    iter_num_per_epoch = len(train_loader)
    Model = Construct_Model(args, input_channel_num, iter_num_per_epoch)
    model, optimizer, scheduler = Model.get_model()
    init_epoch = Model.get_init_epoch()
    Loss = Loss_Fcns(args)
    model.cuda()

    for epoch in range(init_epoch, args.epoch+1):
        train(epoch, train_loader, model, optimizer, scheduler, Loss, tb_writer)
        loss = eval(epoch, eval_loader, model, Loss, tb_writer)
        if epoch == 0:
            if args.save_best:
                threshold = loss*args.init_point
        else:
            if loss<threshold:
                is_best, threshold = True, loss
            file_dealer.save_checkpoint(args, epoch, model, optimizer, scheduler, is_best=is_best)
        is_best = False

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if torch.cuda.is_available() is False:
        raise Exception("This module requires GPU support!!")
    args = Access_Args().get_args()
    main(args)