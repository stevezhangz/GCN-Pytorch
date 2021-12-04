import torch
import numpy as np
import random

def setup_seed(random_number):
    torch.manual_seed(random_number)
    torch.cuda.manual_seed(random_number)
    np.random.seed(random_number)
    random.seed(random_number)
