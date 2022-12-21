import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    """Set random seeds for random, numpy, torch, and torch.backends.cudnn."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def save_model(model, model_path):
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    torch.save(model_state_dict, model_path)

def load_model(model, model_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(model_file, map_location=device))
    print('Network parameters loaded to the {} model.'.format(type(model).__name__))
