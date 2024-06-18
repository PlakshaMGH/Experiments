import random
import torch
import numpy as np


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
