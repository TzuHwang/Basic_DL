import os, sys, natsort, cv2
import torch
import numpy as np

from arguments import Access_Args
from dataset import Data_Loader
from models import Construct_Model
from loss import Loss_Fcns
from tb_visualization import Result_Visualization
from finale import Finale
from utils import file_dealer, tb_lib, pyg_support, misc


def train(args, epoch, train_loader, model, optimizer, scheduler, loss, writer):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = misc.to_model(data, args.data_format), target.cuda()

        if args.task in ["segmentation", "classification", "detection"]:
            pred = model(input)
            loss_matrix = loss.get_loss_value(pred, target)
            loss_matrix['loss'].backward()
            optimizer.step()
            scheduler.step()

            # write log
            input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()
            # Epoch starts at 1. This feature sets initial iter_num to loader size. To start iter_num at 0, subtract 1 from initial epoch.
            iter_num = (epoch-1) * len(train_loader) + i  
            if iter_num % args.print_frequency == 0:
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
        elif args.task in ["multitask"]:
            pred = model(input)
            loss_matrix = loss.get_loss_value(pred, target)
            loss_matrix['loss'].backward()
            optimizer.step()
            scheduler.step()


def eval(args, epoch, eval_loader, model, loss, writer):
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(eval_loader):
            input, target = misc.to_model(data, args.data_format), target.cuda()

            epoch_loss = 0
            if args.task in ["segmentation", "classification", "detection"]:
                pred = model(input)
                loss_matrix = loss.get_loss_value(pred, target)
                epoch_loss += loss_matrix['loss']

                # write log
                input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()
                iter_num = (epoch-1) * len(eval_loader) + i
                if iter_num % args.print_frequency == 0:
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
        return epoch_loss/len(eval_loader)

def main(args):
    if args.section in ["training", "final_val"]:
        train_loader = Data_Loader(args, 'train').get_loader()
        eval_loader = Data_Loader(args, 'val').get_loader()
    elif args.section in ["test", "final_test"]:
        train_loader = Data_Loader(args, 'train_final').get_loader()
        eval_loader = Data_Loader(args, 'test').get_loader()
        
    print(f"Train data sizs:{len(train_loader)}")
    print(f"Eval data sizs:{len(eval_loader)}")

    assert(args.print_frequency<=len(eval_loader))
    tb_writer = Result_Visualization(args)
    input_channel_num = misc.get_input_channel_num(train_loader, args.data_format)
    iter_num_per_epoch = len(train_loader)
    Model = Construct_Model(args, input_channel_num, iter_num_per_epoch)
    model, optimizer, scheduler = Model.get_model()
    init_epoch = Model.get_init_epoch()
    Loss = Loss_Fcns(args)
    model.cuda()

    if args.section in ["training", "test"]:
        for epoch in range(init_epoch, args.epoch+1):
            train(args, epoch, train_loader, model, optimizer, scheduler, Loss, tb_writer)
            loss = eval(args, epoch, eval_loader, model, Loss, tb_writer)
            if epoch == init_epoch:
                if args.save_best:
                    threshold = loss*args.init_point
            else:
                if loss<threshold:
                    is_best, threshold = True, loss
                file_dealer.save_checkpoint(args, epoch, model, optimizer, scheduler, is_best=is_best)
            is_best = False
    elif args.section in ["final_test", "final_val"]:
        Finale(args, eval_loader, model)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if torch.cuda.is_available() is False:
        raise Exception("This module requires GPU support!!")
    args = Access_Args().get_args()
    main(args)