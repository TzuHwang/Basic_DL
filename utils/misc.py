import random
import numpy as np

def roll_a_dice(range = [1, 100], normalize=True):
    score = random.randint(range[0], range[1])
    score = score / np.max(range) if normalize else score
    return score

if __name__ == "__main__":
    iter_num = 10000
    scores = np.array([roll_a_dice() for i in range(iter_num)])
    print(np.sum(scores>0.8)/iter_num)
    print(np.sum(scores<=0.8)/iter_num)