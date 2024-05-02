import numpy as np
from matplotlib import pyplot as plt


def Sample():
    return -1 + 2 * np.random.random((2,))

def PlotVec(vec):
    origin = (0,0)
    plt.plot(*zip(origin, vec), c="blue")
    return 

def FormatPlot(hard_limit):
    plt.xlim(-hard_limit, hard_limit)
    plt.ylim(-hard_limit, hard_limit)
    plt.title("fancy noise")
    plt.xticks([])
    plt.yticks([])
    return 
    

if __name__ == "__main__":
    plt.ion()
    cur_noise = np.zeros((2,))
    decay_rate = 0.9
    hard_limit = 5.0
    while True:
        cur_noise += Sample()
        cur_noise.clip(-hard_limit, hard_limit)
        cur_noise *= decay_rate
        plt.clf()
        FormatPlot(hard_limit)
        PlotVec(cur_noise)
        plt.pause(0.1)
        
