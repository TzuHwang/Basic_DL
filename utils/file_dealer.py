import os, sys, natsort, json, random, cv2, torch
import numpy as np

def img_loader(pth, keep_color=False):
    ext = pth.split('.')[-1]
    if ext in ['npy']:
        image = np.load(pth).astype(np.int16)
    elif ext in ['png', 'jpg']:
        image = cv2.imread(pth) if keep_color else cv2.imread(pth, 0)
    else:
        raise Exception("Unspporting image format")
    return image

def json_loader(fpth):
    with open(fpth) as f:
        data = json.load(f)
    return data

def json_saver(fpth, jsdata):
    with open(fpth, 'w') as f:
        json.dump(jsdata, f, indent=2)

def save_checkpoint(args, epoch, model, optimizer, scheduler, is_best=False):
    save_pth = f"{args.output_root}/{args.config_name}/epoch_{epoch}.pth"
    if epoch % args.save_frequency==0 or is_best:
        ckpt = {
            "args":args,
            "epoch": epoch,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),
            }
        torch.save(ckpt, save_pth)