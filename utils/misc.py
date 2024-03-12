import random
import numpy as np

def roll_a_dice(range = [1, 100], normalize=True):
    score = random.randint(range[0], range[1])
    score = score / np.max(range) if normalize else score
    return score

def get_input_channel_num(dataloader, data_format):
    if data_format == "graph":
        input_channel_num = dataloader.dataset[0][0].x.shape[-1]
    else:
        input_channel_num = dataloader.dataset[0][0].shape[0]
    return input_channel_num

def to_model(data, data_format):
    if data_format == "graph":
        data = data.cuda()
    else:
        data = data.cuda().float()
    return data

if __name__ == "__main__":
    iter_num = 10000
    scores = np.array([roll_a_dice() for i in range(iter_num)])
    print(np.sum(scores>0.8)/iter_num)
    print(np.sum(scores<=0.8)/iter_num)