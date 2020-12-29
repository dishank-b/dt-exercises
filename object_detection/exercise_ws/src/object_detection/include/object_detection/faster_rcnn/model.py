import torch
import numpy as np
from src.architecture import FasterRCNN


def create_model(checkpoint_path):
    # print("Using Model {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    cfg = checkpoint['cfg']

    device = torch.device("cuda") if (torch.cuda.is_available() and cfg.USE_CUDA) else torch.device("cpu")
    print("Using the device for training: {} \n".format(device))

    model = FasterRCNN(cfg)
    model = model.to(device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()

    return model
